# src/model/grm_v2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class LightweightEncoder(nn.Module):
    """保持与原版相同"""

    def __init__(self, in_channels=128, hidden_channels=128):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, hidden_channels, 1)
        self.blocks = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1,
                      groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        x = self.conv_in(x)
        return x + self.blocks(x)


class CrossAttentionConditionalDecoder(nn.Module):
    """
    基于交叉注意力的条件解码器
    用 prompt_seq 作为 K, V 来引导特征修复
    """

    def __init__(self, feat_channels=128, prompt_dim=128, num_heads=4):
        super().__init__()
        self.feat_channels = feat_channels
        self.prompt_dim = prompt_dim
        self.num_heads = num_heads

        assert feat_channels % num_heads == 0, "feat_channels must be divisible by num_heads"

        # Query: 来自特征图
        self.query_proj = nn.Conv2d(feat_channels, feat_channels, 1)

        # Key & Value: 来自 prompt_seq
        self.key_proj = nn.Linear(prompt_dim, feat_channels)
        self.value_proj = nn.Linear(prompt_dim, feat_channels)

        # 输出投影
        self.out_proj = nn.Conv2d(feat_channels, feat_channels, 1)

        # 后续的卷积块
        self.refine_blocks = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, 1, 1,
                      groups=feat_channels, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.scale = (feat_channels // num_heads) ** -0.5

    def forward(
            self,
            feature: torch.Tensor,  # (B, C, H, W)
            prompt_seq: torch.Tensor  # (B, N, D)
    ) -> torch.Tensor:
        """
        使用 prompt_seq 通过交叉注意力引导特征修复
        """
        B, C, H, W = feature.shape
        _, N, D = prompt_seq.shape

        # 1. 准备 Q, K, V
        Q = self.query_proj(feature)  # (B, C, H, W)
        Q = Q.flatten(2).transpose(1, 2)  # (B, H*W, C)

        K = self.key_proj(prompt_seq)  # (B, N, C)
        V = self.value_proj(prompt_seq)  # (B, N, C)

        # 2. Multi-head attention
        # Reshape for multi-head: (B, num_heads, L, head_dim)
        head_dim = C // self.num_heads

        Q = Q.view(B, H * W, self.num_heads, head_dim).transpose(1, 2)  # (B, nh, HW, hd)
        K = K.view(B, N, self.num_heads, head_dim).transpose(1, 2)  # (B, nh, N, hd)
        V = V.view(B, N, self.num_heads, head_dim).transpose(1, 2)  # (B, nh, N, hd)

        # 3. Attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, nh, HW, N)
        attn = F.softmax(attn, dim=-1)

        # 4. Weighted sum
        out = torch.matmul(attn, V)  # (B, nh, HW, hd)

        # 5. Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, H * W, C)  # (B, HW, C)
        out = out.transpose(1, 2).view(B, C, H, W)  # (B, C, H, W)

        # 6. Output projection
        out = self.out_proj(out)

        # 7. Residual + Refinement
        out = feature + out
        out = out + self.refine_blocks(out)

        return out


class VQFeatureRefinementModule_V2(nn.Module):
    """
    升级版 GRM 模块
    - 移除内部 VQ (由 SPM 负责)
    - 使用交叉注意力解码器
    """

    def __init__(
            self,
            channels=128,
            prompt_dim=128,
            num_heads=4
    ):
        super().__init__()
        self.channels = channels
        self.prompt_dim = prompt_dim

        self.encoder = LightweightEncoder(channels, channels)

        # 核心改动: 使用交叉注意力解码器
        self.decoder = CrossAttentionConditionalDecoder(
            feat_channels=channels,
            prompt_dim=prompt_dim,
            num_heads=num_heads
        )

        print(f"✅ GRM-V2 initialized with Cross-Attention decoder")
        print(f"   - Feature channels: {channels}")
        print(f"   - Prompt dim: {prompt_dim}")
        print(f"   - Num attention heads: {num_heads}")

    def forward(
            self,
            feature: torch.Tensor,  # (B, C, H, W)
            prompt_seq: torch.Tensor  # (B, N, D)
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            feature: 待修复的特征图
            prompt_seq: 来自 SPM 的离散化提示序列

        Returns:
            feature_delta: 特征修复残差
            info_dict: 空字典 (为了保持接口兼容)
        """
        # 1. 编码
        z = self.encoder(feature)

        # 2. 交叉注意力解码
        decoded = self.decoder(z, prompt_seq)

        # 3. 计算残差
        feature_delta = decoded - feature

        # 不再有 VQ loss (由 SPM 负责)
        info_dict = {}

        return feature_delta, info_dict

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        print("✅ GRM-V2 has been frozen (plug-and-play mode).")

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        print("✅ GRM-V2 has been unfrozen (fine-tuning mode).")