# 最小可用示例：把增强器插到 YOLOv8n 上做预测或训练
from pathlib import Path

import torch

from src.yolo_plugin.attach_enhancer import attach_enhancer_to_yolov8, get_last_enhancer_info
from ultralytics import YOLO

print("Using device:", "CUDA" if torch.cuda.is_available() else "CPU")


def main():
    # 1) 加载 YOLOv8n（可用 .pt 权重或 .yaml）
    yolo = YOLO("yolov8n.pt")  # 或 YOLO("yolov8n.yaml")

    # 2) 预训练权重路径（使用 pathlib，避免转义问题）
    pretrained_path = Path(r"D:\code_data\python\ultralytics-main\checkpoints\pretrain_v2\enhancer_v2_best.pt")

    # 3) 挂载增强器（默认冻结，即插即用）
    attach_enhancer_to_yolov8(
        yolo,
        img_size=640,
        pretrained_path=str(pretrained_path),  # 传入 str(pretrained_path) 或直接传 Path 也可
        freeze=True,
    )

    # 4) 预测
    yolo.predict(source=r"D:\code_data\python\ultralytics-main\ultralytics\assets\bus.jpg", imgsz=640, conf=0.25)
    info = get_last_enhancer_info(yolo)
    if info is not None:
        print("Enhancer info keys:", list(info.keys()))
        if info.get("prompt_indices") is not None:
            print("Prompt indices shape:", tuple(info["prompt_indices"].shape))
        print("Condition logits shape:", tuple(info["condition_logits"].shape))

    # 5) 训练（示例：coco128）
    # yolo.train(data="coco128.yaml", imgsz=640, epochs=10, batch=16)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
