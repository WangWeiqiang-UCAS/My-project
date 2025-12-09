from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """将图像划分为patch并嵌入."""

    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 使用卷积实现patch embedding
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.proj(x)
        # (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        _B, _C, _H, _W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x


class StatisticalEncoder(nn.Module):
    """保持与原版相同."""

    def __init__(self, output_dim=64):
        super().__init__()
        stat_dim = 15
        self.stat_mlp = nn.Sequential(
            nn.Linear(stat_dim, 128), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(128, output_dim)
        )

    def compute_statistics(self, image: torch.Tensor) -> torch.Tensor:
        B, C, _H, _W = image.shape
        pixels = image.view(B, C, -1)
        mean, std = pixels.mean(dim=-1), pixels.std(dim=-1)
        min_val, max_val = pixels.min(dim=-1)[0], pixels.max(dim=-1)[0]

        if C == 3:
            luminance = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
        else:
            luminance = pixels[:, 0]

        skewness = ((luminance - luminance.mean(dim=-1, keepdim=True)) ** 3).mean(dim=-1) / (
            luminance.std(dim=-1, unbiased=False) ** 3 + 1e-6
        )
        kurtosis = ((luminance - luminance.mean(dim=-1, keepdim=True)) ** 4).mean(dim=-1) / (
            luminance.std(dim=-1, unbiased=False) ** 4 + 1e-6
        )
        dark_ratio = (luminance < 0.2).float().mean(dim=-1)

        return torch.cat(
            [mean, std, min_val, max_val, skewness.unsqueeze(-1), kurtosis.unsqueeze(-1), dark_ratio.unsqueeze(-1)],
            dim=-1,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        stats = self.compute_statistics(image)
        return self.stat_mlp(stats)


class PromptSequenceVQ(nn.Module):
    """对提示序列进行向量量化的模块 将连续的 patch 特征映射到离散的提示码本.
    """

    def __init__(self, num_embeddings=512, embedding_dim=128, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # 提示码本
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        # EMA 更新
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())
        self.register_buffer("code_usage", torch.zeros(num_embeddings))
        self.register_buffer("batch_count", torch.zeros(1))
        self.decay = 0.99

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            z: (B, N, D) 连续的 patch 特征序列

        Returns:
            quantized: (B, N, D) 量化后的序列
            indices: (B, N) 每个 patch 对应的码本索引
            info: 包含损失和统计信息的字典.
        """
        original_dtype = z.dtype
        z_float32 = z.float()

        B, N, D = z_float32.shape

        # ✅ 修复：使用 reshape 或先调用 contiguous()
        # 方法1：使用 reshape（推荐，自动处理连续性）
        z_flat = z_float32.reshape(-1, D)  # (B*N, D)

        # 或者方法2：先确保连续性再 view
        # z_flat = z_float32.contiguous().view(-1, D)

        # 计算到所有码本向量的距离
        distances = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )

        # 找到最近的码本索引
        encoding_indices = torch.argmin(distances, dim=1)
        indices = encoding_indices.view(B, N)  # 这里可以用 view，因为是一维展平

        # 量化
        quantized_flat = self.embedding(encoding_indices)

        # ✅ 修复：使用 reshape
        quantized = quantized_flat.reshape(B, N, D)

        # 计算 VQ 损失
        if self.training:
            e_latent_loss = F.mse_loss(quantized.detach(), z_float32)
            q_latent_loss = F.mse_loss(quantized, z_float32.detach())
            vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

            # EMA 更新
            with torch.no_grad():
                self.code_usage.add_(torch.bincount(encoding_indices, minlength=self.num_embeddings))
                self.batch_count.add_(1)

                encoding_onehot = F.one_hot(encoding_indices, self.num_embeddings).float()
                encodings_sum = encoding_onehot.sum(0)
                dw = torch.matmul(encoding_onehot.t(), z_flat)

                self.ema_cluster_size.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

                n = self.ema_cluster_size.sum()
                cluster_size = (self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
                self.embedding.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))

                # 码本复活机制
                if self.batch_count % 100 == 0:
                    dead_codes = (self.code_usage < 1).nonzero(as_tuple=True)[0]
                    if len(dead_codes) > 0 and z_flat.size(0) >= len(dead_codes):
                        alive_indices = torch.randperm(z_flat.size(0), device=z.device)[: len(dead_codes)]

                        self.embedding.weight.data[dead_codes] = z_flat[alive_indices]
                        self.ema_w.data[dead_codes] = z_flat[alive_indices]
                        self.ema_cluster_size[dead_codes] = 1.0
                        self.code_usage.zero_()
        else:
            vq_loss = torch.tensor(0.0, device=z.device, dtype=torch.float32)

        # Straight-through estimator
        quantized = z_float32 + (quantized - z_float32).detach()
        quantized = quantized.to(original_dtype)

        # 计算 perplexity
        with torch.no_grad():
            avg_probs = F.one_hot(encoding_indices, self.num_embeddings).float().mean(0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        info = {"vq_loss": vq_loss, "perplexity": perplexity, "unique_codes": len(torch.unique(indices))}

        return quantized, indices, info


class ScenePromptModule_V2(nn.Module):
    """升级版场景提示模块 输出: 离散化的 patch-level 提示序列.
    """

    def __init__(
        self,
        input_channels=3,
        img_size=256,
        patch_size=16,
        prompt_dim=128,
        num_conditions=8,
        vq_num_embeddings=512,
        vq_commitment_cost=0.25,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.prompt_dim = prompt_dim
        self._init_img_size = img_size
        self._init_grid_size = img_size // patch_size

        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, in_channels=input_channels, embed_dim=prompt_dim
        )

        # 2. 全局统计编码器
        self.stat_encoder = StatisticalEncoder(output_dim=prompt_dim)

        # 3. 位置编码 (保存原始 grid 大小以便插值)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, prompt_dim) * 0.02)

        # 4. 轻量级 Transformer Encoder (可选，用于融合局部和全局信息)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=prompt_dim,
                nhead=4,
                dim_feedforward=prompt_dim * 2,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2,
        )

        # 5. 向量量化层 (核心创新)
        self.vq_layer = PromptSequenceVQ(
            num_embeddings=vq_num_embeddings, embedding_dim=prompt_dim, commitment_cost=vq_commitment_cost
        )

        # 6. 多标签分类头 (用于退化类型识别)
        self.multi_label_head = nn.Linear(prompt_dim, num_conditions)

        print("✅ SPM-V2 initialized:")
        print(f"   - Image size: {img_size}x{img_size}")
        print(f"   - Patch size: {patch_size}x{patch_size}")
        print(f"   - Num patches: {self.num_patches}")
        print(f"   - Prompt dim: {prompt_dim}")
        print(f"   - VQ codebook size: {vq_num_embeddings}")

    def _interpolated_pos_embed(
        self, device: torch.device, dtype: torch.dtype, target_h: int, target_w: int
    ) -> torch.Tensor:
        """将 self.pos_embed 从初始化的 (1, H0*W0, D) 插值到 (1, target_h*target_w, D)."""
        # reshape to [1, D, H0, W0]
        D = self.pos_embed.shape[-1]
        H0 = self._init_grid_size
        W0 = self._init_grid_size
        pos = self.pos_embed.view(1, H0, W0, D).permute(0, 3, 1, 2)  # [1, D, H0, W0]
        pos = pos.to(device=device, dtype=dtype)
        # use bicubic interpolation for smoothness
        pos_interp = F.interpolate(pos, size=(target_h, target_w), mode="bicubic", align_corners=False)
        pos_interp = pos_interp.permute(0, 2, 3, 1).reshape(1, target_h * target_w, D)
        return pos_interp

    def forward(self, image: torch.Tensor, use_vq: bool = True) -> dict:
        """
        Args:
            image: (B, 3, H, W)
            use_vq: 是否使用向量量化.

        Returns:
            {
            'prompt_seq': (B, N, D) 离散化的提示序列,
            'prompt_indices': (B, N) 码本索引 (如果use_vq=True),
            'condition_logits': (B, num_conditions) 多标签分类logits,
            'vq_loss': scalar,
            'perplexity': scalar, ... }
        """
        image.size(0)
        _, _, H_img, W_img = image.shape

        # 1. Patch embedding
        patch_seq = self.patch_embed(image)  # (B, N, D)
        _B, _N, _D = patch_seq.shape

        # determine current grid size
        cur_h = H_img // self.patch_size
        cur_w = W_img // self.patch_size
        cur_num_patches = cur_h * cur_w

        if cur_num_patches != self.num_patches:
            # 插值/适配位置编码到当前网格
            if H_img % self.patch_size != 0 or W_img % self.patch_size != 0:
                # 非整除边界情况：提示并继续（也可以在此对图像做 pad 或 resize）
                print(
                    f"⚠️ 输入图像大小 ({H_img}x{W_img}) 不能被 patch_size ({self.patch_size}) 整除。"
                    f" 将按整除下取整的网格 ({cur_h}x{cur_w}) 处理。"
                )
            pos = self._interpolated_pos_embed(
                device=patch_seq.device, dtype=patch_seq.dtype, target_h=cur_h, target_w=cur_w
            )
        else:
            pos = self.pos_embed.to(device=patch_seq.device, dtype=patch_seq.dtype)

        # 2. 添加位置编码（广播设备/dtype 已对齐）
        prompt_seq = patch_seq + pos

        # 3. 全局统计特征
        stat_feat = self.stat_encoder(image)  # (B, D)

        # 将全局特征广播到每个 patch (简单的融合策略)
        prompt_seq = prompt_seq + stat_feat.unsqueeze(1)

        # 4. Transformer 编码 (融合上下文)
        prompt_seq = self.transformer(prompt_seq)  # (B, N, D)

        # 5. 向量量化 (关键步骤)
        vq_info = {}
        if use_vq:
            quantized_seq, indices, vq_info = self.vq_layer(prompt_seq)
        else:
            quantized_seq = prompt_seq
            indices = None

        # 6. 多标签分类 (使用全局池化)
        global_feat = quantized_seq.mean(dim=1)  # (B, D)
        condition_logits = self.multi_label_head(global_feat)

        return {
            "prompt_seq": quantized_seq,  # (B, N, D)
            "prompt_indices": indices,  # (B, N) or None
            "condition_logits": condition_logits,
            "global_feat": global_feat,
            **vq_info,  # vq_loss, perplexity, unique_codes
        }

    def manipulate_prompt(
        self,
        prompt_indices: torch.Tensor,
        operation: str = "replace",
        target_code: int | None = None,
        source_code: int | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """准符号化操作: 直接操纵提示序列的离散编码.

        Args:
            prompt_indices: (B, N) 当前的码本索引
            operation: 'replace', 'add', 'remove'
            target_code: 要替换成的目标码
            source_code: 要被替换的源码 (仅用于 replace)
            positions: 要操作的位置 (B, N) bool mask

        Returns:
            new_prompt_indices: (B, N) 修改后的索引
        """
        new_indices = prompt_indices.clone()

        if positions is None:
            positions = torch.ones_like(new_indices, dtype=torch.bool)

        if operation == "replace" and source_code is not None and target_code is not None:
            # 将特定的源码替换为目标码
            mask = (new_indices == source_code) & positions
            new_indices[mask] = target_code

        elif operation == "add" and target_code is not None:
            # 在指定位置添加(覆盖)目标码
            new_indices[positions] = target_code

        elif operation == "remove" and source_code is not None:
            # 移除特定码 (这里用一个中性码代替，例如0)
            mask = (new_indices == source_code) & positions
            new_indices[mask] = 0

        return new_indices

    def reconstruct_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """从离散索引重建提示序列.

        Args:
            indices: (B, N)

        Returns:
            prompt_seq: (B, N, D)
        """
        return self.vq_layer.embedding(indices)
