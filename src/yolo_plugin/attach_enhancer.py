"""
Attach GeneralPurposeEnhancer_V2 to a YOLOv8 model (Detect head) with optional profiling.

Features:
- Register enhancer as a submodule so it moves with the model (.to(), .half(), .eval(), ...)
- Cache the input image via a forward_pre_hook for SPM consumption
- Monkey-patch Detect.forward to run enhancer before original Detect
- Lazy device/dtype sync for the enhancer on first real batch
- Optional fine-grained profiling of SPM / GRM (per-layer) / Detect forward times
- Exposes helper APIs on yolo: yolo.enhancer, yolo.enhancer_state, get_last_enhancer_info, get_last_enhancer_profiling

Usage:
    from ultralytics import YOLO
    from src.yolo_plugin.attach_enhancer import attach_enhancer_to_yolov8
    yolo = YOLO("yolov8n.pt")
    attach_enhancer_to_yolov8(yolo, img_size=640, pretrained_path=None, freeze=True, enable_profiling=True)
    yolo.predict(...)
"""

from __future__ import annotations

import time
import types
from typing import Any

import torch
import torch.nn as nn

try:
    # Ultralytics 8.x typing hint only
    from ultralytics.engine.model import YOLO  # type: ignore
except Exception:
    YOLO = None  # type: ignore

from src.model.enhancer_v2 import build_enhancer_for_yolo


def _find_detect_module(root: nn.Module) -> nn.Module:
    for m in root.modules():
        if m.__class__.__name__ == "Detect":
            return m
    raise RuntimeError("未找到 Detect 模块。请确认是 YOLOv8 检测模型。")


def _first_layer(model) -> nn.Module:
    # Ultralytics DetectionModel 通常有 model: nn.ModuleList
    if hasattr(model, "model") and isinstance(model.model, (nn.ModuleList, list)) and len(model.model) > 0:
        return model.model[0]
    for m in model.modules():
        if len(list(m.children())) == 0:
            return m
    raise RuntimeError("未能定位模型的第一层。")


def _sync_module_device_dtype(module: nn.Module, ref: torch.Tensor):
    """Move module to same device/dtype as ref (only when necessary)."""
    p = None
    for param in module.parameters():
        p = param
        break
    if p is None:
        for _, buf in module.named_buffers(recurse=True):
            p = buf
            break
    if p is None:
        return

    need_move = p.device != ref.device
    need_cast = p.dtype != ref.dtype

    if need_move and need_cast:
        module.to(device=ref.device, dtype=ref.dtype)
    elif need_move:
        module.to(device=ref.device)
    elif need_cast:
        module.to(dtype=ref.dtype)


class YOLOEnhancerState:
    def __init__(self):
        self.cached_image: torch.Tensor | None = None
        self.last_info: dict[str, Any] | None = None
        self.synced: bool = False
        self.last_profiling: dict[str, Any] | None = None


def attach_enhancer_to_yolov8(
    yolo: YOLO,
    img_size: int = 640,
    pretrained_path: str | None = None,
    freeze: bool = True,
    verbose: bool = True,
    enable_profiling: bool = True,
    profiling_warmup: int = 1,
):
    """Attach the enhancer to a loaded YOLO model.

    Args:
        yolo: YOLO instance
        img_size: enhancer img_size (for SPM pos_embed shape)
        pretrained_path: optional enhancer checkpoint
        freeze: if True, call enhancer.freeze()
        verbose: print status
        enable_profiling: if True, collect and print per-stage timing (SPM/GRM/Detect)
        profiling_warmup: number of warmup iterations to skip in profiling averages
    """
    model = yolo.model
    detect = _find_detect_module(model)
    state = YOLOEnhancerState()

    # 1) build enhancer (do NOT .to(device) here; register as submodule instead)
    enhancer = build_enhancer_for_yolo(img_size=img_size, pretrained_path=pretrained_path, freeze=freeze)

    # register as a child module so .to() and .state_dict() propagate
    model.add_module("_pgrm_enhancer", enhancer)

    # 2) pre-hook to cache original input image
    def cache_input_image(_module, inputs):
        if inputs and isinstance(inputs[0], torch.Tensor):
            state.cached_image = inputs[0]
        return None

    first = _first_layer(model)
    first.register_forward_pre_hook(cache_input_image)

    # 3) monkey-patch Detect.forward
    orig_forward = detect.forward

    def enhanced_detect_forward(self, features: list[torch.Tensor], *args, **kwargs):
        """Replaces Detect.forward to: - optionally profile SPM and GRM - call enhancer to get enhanced features - pass
        enhanced features to original Detect.forward.
        """
        # If no cached image (very unlikely), just call original forward
        if state.cached_image is None:
            return orig_forward(features, *args, **kwargs)

        # Ensure enhancer device/dtype sync (lazy)
        if not state.synced:
            try:
                _sync_module_device_dtype(enhancer, state.cached_image)
            except Exception:
                # best-effort; ignore sync errors
                pass
            state.synced = True
        else:
            # occasional guard if runtime changed devices/dtypes
            try:
                p = next(enhancer.parameters())
                if p.device != state.cached_image.device or p.dtype != state.cached_image.dtype:
                    _sync_module_device_dtype(enhancer, state.cached_image)
            except Exception:
                pass

        device = state.cached_image.device
        use_cuda = device.type == "cuda"

        def _sync_cuda():
            if use_cuda:
                torch.cuda.synchronize()

        profiling = {
            "spm_ms": None,
            "grm_ms": None,
            "grm_per_layer_ms": [],
            "detect_ms": None,
            "total_ms": None,
        }

        # If profiling enabled, measure SPM and GRM separately.
        try:
            if enable_profiling:
                # Warmup handling (simple): if we haven't profiled yet, skip first N iterations as warmup
                if state.last_profiling is None:
                    state._profiling_runs = 0  # type: ignore
                else:
                    state._profiling_runs = state._profiling_runs + 1 if hasattr(state, "_profiling_runs") else 0

                # 1) profile SPM
                _sync_cuda()
                t0 = time.time()
                spm_out = enhancer.spm(state.cached_image, use_vq=True)
                _sync_cuda()
                t_spm = (time.time() - t0) * 1000.0
                profiling["spm_ms"] = t_spm

                # 2) profile GRM per layer
                _sync_cuda()
                t0 = time.time()
                grm_layer_times = []
                enhanced_feats = []
                for i, fmap in enumerate(features):
                    # time in-proj
                    t_in0 = time.time()
                    proj = enhancer.grm_in_projs[i](fmap)
                    _sync_cuda()
                    t_in1 = time.time()

                    # time encoder+cross-attention+decoder
                    t_encdec0 = time.time()
                    delta, _ = enhancer.grm(proj, spm_out["prompt_seq"])
                    _sync_cuda()
                    t_encdec1 = time.time()

                    # time out-proj
                    t_out0 = time.time()
                    refined = enhancer.grm_out_projs[i](delta)
                    _sync_cuda()
                    t_out1 = time.time()

                    in_proj_ms = (t_in1 - t_in0) * 1000.0
                    encdec_ms = (t_encdec1 - t_encdec0) * 1000.0
                    out_proj_ms = (t_out1 - t_out0) * 1000.0
                    total_layer_ms = in_proj_ms + encdec_ms + out_proj_ms
                    grm_layer_times.append(
                        {
                            "layer": i,
                            "in_proj_ms": in_proj_ms,
                            "encdec_ms": encdec_ms,
                            "out_proj_ms": out_proj_ms,
                            "total_layer_ms": total_layer_ms,
                        }
                    )
                    enhanced_feats.append(fmap + refined)

                _sync_cuda()
                t_grm = (time.time() - t0) * 1000.0
                profiling["grm_ms"] = t_grm
                profiling["grm_per_layer_ms"] = grm_layer_times

                # 3) time original Detect.forward with enhanced_feats
                _sync_cuda()
                t0 = time.time()
                res = orig_forward(enhanced_feats, *args, **kwargs)
                _sync_cuda()
                t_detect = (time.time() - t0) * 1000.0
                profiling["detect_ms"] = t_detect
                profiling["total_ms"] = profiling["spm_ms"] + profiling["grm_ms"] + profiling["detect_ms"]

                # store profiling to state
                state.last_profiling = profiling

                # If still warming up, avoid printing detailed times until warmup passes
                warmup_runs = profiling_warmup
                runs = getattr(state, "_profiling_runs", 0)
                if runs >= warmup_runs:
                    # Print a compact summary
                    try:
                        print(
                            f"[Enhancer Profiling] SPM={profiling['spm_ms']:.1f}ms GRM={profiling['grm_ms']:.1f}ms "
                            f"Detect={profiling['detect_ms']:.1f}ms Total≈{profiling['total_ms']:.1f}ms"
                        )
                        for lt in profiling["grm_per_layer_ms"]:
                            print(
                                f"  Layer{lt['layer']}: in_proj={lt['in_proj_ms']:.1f}ms encdec={lt['encdec_ms']:.1f}ms out_proj={lt['out_proj_ms']:.1f}ms total={lt['total_layer_ms']:.1f}ms"
                            )
                    except Exception:
                        pass
                return res
            else:
                # Non-profiling fast path: call enhancer once and forward to detect
                enhanced_feats, info = enhancer(state.cached_image, features)
                state.last_info = info
                return orig_forward(enhanced_feats, *args, **kwargs)

        except Exception as e:
            # In case enhancer fails for any reason, fallback to original detect to avoid total crash
            if verbose:
                print(f"[Enhancer] Error during enhanced forward: {e}. Falling back to original Detect.forward.")
            try:
                return orig_forward(features, *args, **kwargs)
            except Exception:
                # if even original fails, re-raise
                raise

    # bind the monkey patched method
    detect.forward = types.MethodType(enhanced_detect_forward, detect)

    # attach helper attributes
    yolo.enhancer = enhancer
    yolo.enhancer_state = state
    yolo.enhancer_attached = True

    if verbose:
        what = "冻结(即插即用)" if freeze else "可训练(微调)"
        prof = "with profiling" if enable_profiling else "no profiling"
        print(f"[Enhancer] 已挂载到 YOLOv8 Detect 头之前，模式：{what}，img_size={img_size}, {prof}")

    return yolo


def get_last_enhancer_info(yolo: YOLO) -> dict[str, Any] | None:
    state: YOLOEnhancerState = getattr(yolo, "enhancer_state", None)
    return None if state is None else state.last_info


def get_last_enhancer_profiling(yolo: YOLO) -> dict[str, Any] | None:
    state: YOLOEnhancerState = getattr(yolo, "enhancer_state", None)
    return None if state is None else state.last_profiling


def set_enhancer_trainable(yolo: YOLO, trainable: bool = False):
    enh: nn.Module = getattr(yolo, "enhancer", None)
    if enh is None:
        raise RuntimeError("尚未 attach_enhancer_to_yolov8。")
    if trainable:
        enh.unfreeze()
    else:
        enh.freeze()
