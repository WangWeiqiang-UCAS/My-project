# src/model/enhancer_v2.py
# ✅ 即插即用的通用特征增强器
# 作者: WangWeiqiang-UCAS
# 日期: 2025-11-07

"""
GeneralPurposeEnhancer_V2: 即插即用的特征增强模块

设计理念:
    - 输入: 原始图像 + 任意骨干网络输出的特征金字塔
    - 输出: 增强后的特征金字塔 (相同shape)
    - 黑盒操作: 无需了解内部的 SPM/GRM/VQ 实现细节

使用场景:
    1. 目标检测: YOLO/Faster-RCNN/DETR 等
    2. 语义分割: FCN/UNet/DeepLab 等
    3. 实例分割: Mask-RCNN/SOLO 等
    4. 图像恢复: 去噪/去雨/去雾 等

示例:
    >>> enhancer = GeneralPurposeEnhancer_V2(
    ...     feature_channels=[64, 128, 256],  # 骨干网络输出通道
    ...     img_size=640,                      # 输入图像尺寸
    ... )
    >>>
    >>> # 使用
    >>> image = torch.randn(1, 3, 640, 640)
    >>> features = backbone(image)  # [(1,64,H,W), (1,128,H/2,W/2), (1,256,H/4,W/4)]
    >>> enhanced_features, info = enhancer(image, features)
    >>> # enhanced_features 与 features 形状完全相同
"""

from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spm_v2 import ScenePromptModule_V2
from .grm_v2 import VQFeatureRefinementModule_V2


class GeneralPurposeEnhancer_V2(nn.Module):
    """
    通用特征增强器 V2

    核心功能:
        1. 自动分析场景退化类型 (SPM)
        2. 生成离散化的修复指令序列 (VQ)
        3. 通过交叉注意力修复特征 (GRM)
        4. 支持多尺度特征金字塔
        5. 完全即插即用，无需修改主干网络
    """

    def __init__(
            self,
            feature_channels: List[int],
            grm_shared_channels: int = 128,
            prompt_dim: int = 128,
            img_size: int = 256,
            patch_size: int = 16,
            spm_num_conditions: int = 9,  # 支持的退化类型数量
            vq_num_embeddings: int = 512,  # 提示码本大小
            vq_commitment_cost: float = 0.25,
            freeze_grm: bool = False,
            verbose: bool = True
    ):
        """
        初始化增强器

        Args:
            feature_channels: 特征金字塔各层的通道数，例如 [64, 128, 256]
            grm_shared_channels: GRM 内部统一处理的通道数
            prompt_dim: 提示向量的维度
            img_size: 输入图像的尺寸 (假设正方形)
            patch_size: 将图像分成 patch 的大小
            spm_num_conditions: 支持的退化类型数量
            vq_num_embeddings: 离散提示码本的大小
            vq_commitment_cost: VQ 损失的权重
            freeze_grm: 是否冻结 GRM（即插即用模式）
            verbose: 是否打印详细信息
        """
        super().__init__()

        self.feature_channels = feature_channels
        self.grm_shared_channels = grm_shared_channels
        self.prompt_dim = prompt_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.verbose = verbose

        # ========================================
        # 1. 场景分析模块 (SPM)
        # ========================================
        # 功能: 分析图像的退化类型，生成离散化的修复指令
        self.spm = ScenePromptModule_V2(
            input_channels=3,
            img_size=img_size,
            patch_size=patch_size,
            prompt_dim=prompt_dim,
            num_conditions=spm_num_conditions,
            vq_num_embeddings=vq_num_embeddings,
            vq_commitment_cost=vq_commitment_cost
        )

        # ========================================
        # 2. 特征修复模块 (GRM)
        # ========================================
        # 功能: 基于修复指令，通过交叉注意力机制修复特征
        self.grm = VQFeatureRefinementModule_V2(
            channels=grm_shared_channels,
            prompt_dim=prompt_dim,
            num_heads=4
        )

        if freeze_grm:
            self.grm.freeze()

        # ========================================
        # 3. 自适应投影层
        # ========================================
        # 功能: 使增强器能够处理任意通道数的特征
        # 输入投影: 将特征统一到 grm_shared_channels
        self.grm_in_projs = nn.ModuleList([
            nn.Conv2d(in_ch, grm_shared_channels, kernel_size=1, bias=False)
            if in_ch != grm_shared_channels else nn.Identity()
            for in_ch in feature_channels
        ])

        # 输出投影: 将修复后的特征恢复到原始通道数
        self.grm_out_projs = nn.ModuleList([
            nn.Conv2d(grm_shared_channels, out_ch, kernel_size=1, bias=False)
            if out_ch != grm_shared_channels else nn.Identity()
            for out_ch in feature_channels
        ])

        # 初始化投影层权重
        self._init_projection_weights()

        if verbose:
            self._print_info()

    def _init_projection_weights(self):
        """初始化投影层权重"""
        for module in self.grm_in_projs:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

        for module in self.grm_out_projs:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    def _print_info(self):
        """打印模块信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n{'=' * 80}")
        print(f"{'✅ GeneralPurposeEnhancer V2 - 即插即用特征增强器':^80}")
        print(f"{'=' * 80}")
        print(f"\n📋 配置信息:")
        print(f"  输入图像尺寸:        {self.img_size} x {self.img_size}")
        print(f"  Patch 大小:          {self.patch_size} x {self.patch_size}")
        print(f"  Patch 数量:          {(self.img_size // self.patch_size) ** 2}")
        print(f"  特征金字塔通道:      {self.feature_channels}")
        print(f"  GRM 统一通道:        {self.grm_shared_channels}")
        print(f"  提示向量维度:        {self.prompt_dim}")
        print(f"  离散码本大小:        {self.spm.vq_layer.num_embeddings}")
        print(f"  支持退化类型:        {self.spm.multi_label_head.out_features}")

        print(f"\n📊 参数统计:")
        print(f"  总参数量:            {total_params / 1e6:.2f}M")
        print(f"  可训练参数:          {trainable_params / 1e6:.2f}M")
        print(f"  冻结参数:            {(total_params - trainable_params) / 1e6:.2f}M")

        print(f"\n🔧 模块组成:")
        spm_params = sum(p.numel() for p in self.spm.parameters())
        grm_params = sum(p.numel() for p in self.grm.parameters())
        proj_params = (sum(p.numel() for p in self.grm_in_projs.parameters()) +
                       sum(p.numel() for p in self.grm_out_projs.parameters()))
        print(f"  SPM (场景分析):      {spm_params / 1e6:.2f}M ({spm_params / total_params * 100:.1f}%)")
        print(f"  GRM (特征修复):      {grm_params / 1e6:.2f}M ({grm_params / total_params * 100:.1f}%)")
        print(f"  投影层 (自适应):     {proj_params / 1e6:.2f}M ({proj_params / total_params * 100:.1f}%)")

        print(f"\n💡 使用提示:")
        print(f"  1. 输入: image (B,3,H,W) + feature_pyramid [List of Tensors]")
        print(f"  2. 输出: enhanced_features (相同shape) + info (Dict)")
        print(f"  3. 即插即用: 可接入任意骨干网络 (YOLO/ResNet/ViT/...)")
        print(f"  4. 可控编辑: 通过操纵 prompt_indices 实现精确控制")
        print(f"{'=' * 80}\n")

    def forward(
            self,
            image: torch.Tensor,
            feature_pyramid: List[torch.Tensor],
            use_vq: bool = True,
            custom_prompt_indices: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], Dict]:
        """
        前向传播

        Args:
            image: 原始输入图像 (B, 3, H, W)
            feature_pyramid: 骨干网络输出的特征金字塔
                格式: [(B, C1, H1, W1), (B, C2, H2, W2), ...]
            use_vq: 是否使用向量量化（训练时建议True，推理时可选）
            custom_prompt_indices: 自定义的提示码索引 (B, N)
                用于可控编辑，如果提供则使用它而不是自动分析

        Returns:
            enhanced_features: 增强后的特征金字塔 (与输入相同shape)
            info: 辅助信息字典，包含:
                - 'prompt_seq': 提示序列 (B, N, D)
                - 'prompt_indices': 离散码索引 (B, N)
                - 'condition_logits': 退化类型分类 logits (B, num_conditions)
                - 'vq_loss': VQ 损失 (如果 use_vq=True)
                - 'perplexity': 困惑度
                - 'unique_codes': 使用的唯一码数量

        示例:
            >>> enhancer = GeneralPurposeEnhancer_V2([64, 128, 256])
            >>> image = torch.randn(4, 3, 256, 256)
            >>> features = [
            ...     torch.randn(4, 64, 32, 32),
            ...     torch.randn(4, 128, 16, 16),
            ...     torch.randn(4, 256, 8, 8)
            ... ]
            >>> enhanced, info = enhancer(image, features)
            >>> print([f.shape for f in enhanced])
            [(4, 64, 32, 32), (4, 128, 16, 16), (4, 256, 8, 8)]
        """
        # ========================================
        # 阶段 1: 场景分析与提示生成
        # ========================================
        with torch.set_grad_enabled(self.spm.training):
            spm_output = self.spm(image, use_vq=use_vq)

        # 如果提供了自定义索引，则从索引重建提示序列
        if custom_prompt_indices is not None:
            prompt_seq = self.spm.reconstruct_from_indices(custom_prompt_indices)
            prompt_indices = custom_prompt_indices
        else:
            prompt_seq = spm_output['prompt_seq']
            prompt_indices = spm_output.get('prompt_indices')

        # ========================================
        # 阶段 2: 特征修复
        # ========================================
        enhanced_features = []

        for i, feature_map in enumerate(feature_pyramid):
            # 2.1 投影到统一维度
            projected_feat = self.grm_in_projs[i](feature_map)

            # 2.2 GRM 修复 (基于交叉注意力)
            feature_delta, _ = self.grm(projected_feat, prompt_seq)

            # 2.3 投影回原始维度
            refined_delta = self.grm_out_projs[i](feature_delta)

            # 2.4 残差连接
            enhanced_feat = feature_map + refined_delta
            enhanced_features.append(enhanced_feat)

        # ========================================
        # 阶段 3: 收集辅助信息
        # ========================================
        info = {
            'prompt_seq': prompt_seq,
            'prompt_indices': prompt_indices,
            'condition_logits': spm_output['condition_logits'],
            'scene_analysis': spm_output,  # 完整的 SPM 输出
        }

        # 添加 VQ 相关指标
        if 'vq_loss' in spm_output:
            info['vq_loss'] = spm_output['vq_loss']
            info['perplexity'] = spm_output.get('perplexity', torch.tensor(0.0))
            info['unique_codes'] = spm_output.get('unique_codes', 0)

        return enhanced_features, info

    def freeze(self):
        """冻结所有参数（即插即用模式）"""
        for param in self.parameters():
            param.requires_grad = False
        if self.verbose:
            print("🔒 Enhancer V2 已冻结（即插即用模式）")

    def unfreeze(self):
        """解冻所有参数（微调模式）"""
        for param in self.parameters():
            param.requires_grad = True
        if self.verbose:
            print("🔓 Enhancer V2 已解冻（微调模式）")

    def freeze_spm_only(self):
        """仅冻结 SPM（保持场景分析能力，只训练 GRM）"""
        for param in self.spm.parameters():
            param.requires_grad = False
        if self.verbose:
            print("🔒 SPM 已冻结，GRM 保持可训练")

    def freeze_grm_only(self):
        """仅冻结 GRM（保持修复能力，只训练 SPM）"""
        for param in self.grm.parameters():
            param.requires_grad = False
        for proj in self.grm_in_projs:
            for param in proj.parameters():
                param.requires_grad = False
        for proj in self.grm_out_projs:
            for param in proj.parameters():
                param.requires_grad = False
        if self.verbose:
            print("🔒 GRM 已冻结，SPM 保持可训练")

    def get_codebook_usage_stats(self) -> Dict:
        """获取码本使用统计"""
        code_usage = self.spm.vq_layer.code_usage.cpu()
        total_codes = len(code_usage)
        used_codes = (code_usage > 0).sum().item()

        return {
            'total_codes': total_codes,
            'used_codes': used_codes,
            'usage_rate': used_codes / total_codes * 100,
            'code_distribution': code_usage.numpy()
        }

    def manipulate_prompt(
            self,
            original_indices: torch.Tensor,
            operation: str = 'replace',
            target_code: Optional[int] = None,
            source_code: Optional[int] = None,
            positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        准符号化操作: 操纵提示码序列

        Args:
            original_indices: 原始的码本索引 (B, N)
            operation: 操作类型 ('replace', 'add', 'remove')
            target_code: 目标码
            source_code: 源码（仅用于 replace）
            positions: 操作位置的 mask (B, N)

        Returns:
            new_indices: 修改后的索引

        示例:
            >>> # 将左上角区域强制使用"强去雨"策略
            >>> mask = create_topleft_mask(batch_size, num_patches)
            >>> new_indices = enhancer.manipulate_prompt(
            ...     original_indices,
            ...     operation='add',
            ...     target_code=100,  # 假设 100 代表强去雨
            ...     positions=mask
            ... )
        """
        return self.spm.manipulate_prompt(
            original_indices,
            operation=operation,
            target_code=target_code,
            source_code=source_code,
            positions=positions
        )

    @property
    def device(self):
        """返回模型所在设备"""
        return next(self.parameters()).device


# ============================================
# 便捷构建函数
# ============================================

def build_enhancer_for_yolo(
    img_size: int = 640,
    pretrained_path: Optional[str] = None,
    freeze: bool = False
) -> GeneralPurposeEnhancer_V2:
    """
    为 YOLO 系列构建增强器
    
    Args:
        img_size: 输入图像尺寸
        pretrained_path: 预训练权重路径
        freeze: 是否冻结（即插即用）
    """
    enhancer = GeneralPurposeEnhancer_V2(
        feature_channels=[64, 128, 256],
        grm_shared_channels=128,
        prompt_dim=128,
        img_size=img_size,
        patch_size=16,
        spm_num_conditions=9,
        vq_num_embeddings=512
    )
    
    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'enhancer_state_dict' in checkpoint:
            state_dict = checkpoint['enhancer_state_dict']
            
            # ✅ 修复：处理位置编码尺寸不匹配
            current_pos_embed_shape = enhancer.spm.pos_embed.shape  # [1, 1600, 128]
            pretrained_pos_embed_shape = state_dict['spm.pos_embed'].shape  # [1, 256, 128]
            
            if current_pos_embed_shape != pretrained_pos_embed_shape:
                print(f"\n⚠️  位置编码尺寸不匹配:")
                print(f"   预训练: {pretrained_pos_embed_shape}")
                print(f"   当前:   {current_pos_embed_shape}")
                print(f"   🔧 使用插值调整位置编码...\n")
                
                # 插值位置编码
                pretrained_pos_embed = state_dict['spm.pos_embed']  # [1, 256, 128]
                
                # 获取预训练和当前的 patch 数量
                pretrained_num_patches = pretrained_pos_embed_shape[1]  # 256
                current_num_patches = current_pos_embed_shape[1]  # 1600
                
                # 计算网格尺寸
                pretrained_grid_size = int(pretrained_num_patches ** 0.5)  # 16
                current_grid_size = int(current_num_patches ** 0.5)  # 40
                
                # 重塑为 2D 网格
                pretrained_pos_embed = pretrained_pos_embed.reshape(
                    1, pretrained_grid_size, pretrained_grid_size, -1
                ).permute(0, 3, 1, 2)  # [1, 128, 16, 16]
                
                # 双线性插值到新尺寸
                import torch.nn.functional as F
                interpolated_pos_embed = F.interpolate(
                    pretrained_pos_embed,
                    size=(current_grid_size, current_grid_size),
                    mode='bicubic',  # 使用双三次插值（更平滑）
                    align_corners=False
                )  # [1, 128, 40, 40]
                
                # 重塑回 [1, N, D]
                interpolated_pos_embed = interpolated_pos_embed.permute(0, 2, 3, 1).reshape(
                    1, current_num_patches, -1
                )  # [1, 1600, 128]
                
                # 替换 state_dict 中的位置编码
                state_dict['spm.pos_embed'] = interpolated_pos_embed
                
                print(f"   ✅ 位置编码已插值: {pretrained_pos_embed_shape} → {current_pos_embed_shape}\n")
            
            # 加载权重
            enhancer.load_state_dict(state_dict)
            print(f"✅ 已加载预训练权重: {pretrained_path}\n")
        else:
            print(f"⚠️  Checkpoint 中未找到 'enhancer_state_dict'\n")
    
    if freeze:
        enhancer.freeze()
    
    return enhancer

def build_enhancer_for_resnet(
        img_size: int = 224,
        pretrained_path: Optional[str] = None,
        freeze: bool = False
) -> GeneralPurposeEnhancer_V2:
    """为 ResNet 系列构建增强器"""
    enhancer = GeneralPurposeEnhancer_V2(
        feature_channels=[256, 512, 1024],  # ResNet 的 C3, C4, C5
        grm_shared_channels=256,
        prompt_dim=128,
        img_size=img_size,
        patch_size=14,
        spm_num_conditions=9,
        vq_num_embeddings=512
    )

    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'enhancer_state_dict' in checkpoint:
            enhancer.load_state_dict(checkpoint['enhancer_state_dict'])
            print(f"✅ 已加载预训练权重: {pretrained_path}")

    if freeze:
        enhancer.freeze()

    return enhancer