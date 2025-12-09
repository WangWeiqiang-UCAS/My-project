# src/data_process/degradations.py

import torch
import torch.nn.functional as F
import random
import math
from io import BytesIO
from PIL import Image
import numpy as np


# --- 内部辅助函数 (保持不变或新增) ---

def _to_batch(image: torch.Tensor) -> torch.Tensor:
    return image.unsqueeze(0) if image.dim() == 3 else image


def _from_batch(image: torch.Tensor) -> torch.Tensor:
    return image.squeeze(0) if image.dim() == 4 and image.size(0) == 1 else image


def _same_pad_2d(k: int) -> int:
    return k // 2


def _conv2d_channelwise(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    C = img.size(1)
    k = kernel.size(0)
    # 确保 kernel 是 4D
    weight = kernel.view(1, 1, k, k).repeat(C, 1, 1, 1)
    pad = _same_pad_2d(k)
    return F.conv2d(img, weight, bias=None, stride=1, padding=pad, groups=C)


def _make_gaussian_kernel(ks: int, sigma: float) -> torch.Tensor:
    ax = torch.arange(ks, dtype=torch.float32) - (ks - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="xy")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    return kernel


def _make_motion_kernel(length: int, angle_deg: float) -> torch.Tensor:
    length = max(1, int(length))
    k = max(3, length | 1)
    kernel = torch.zeros((k, k), dtype=torch.float32)
    angle = math.radians(angle_deg)
    cx = cy = k // 2
    dx, dy = math.cos(angle), math.sin(angle)
    for t_val in torch.linspace(-length / 2, length / 2, steps=length).tolist():
        x = int(round(cx + t_val * dx))
        y = int(round(cy + t_val * dy))
        if 0 <= x < k and 0 <= y < k:
            kernel[y, x] = 1.0
    s = kernel.sum().clamp_min(1.0)
    return kernel / s


# --- JPEG 压缩的辅助函数 ---
def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.mul(255).byte()
    return Image.fromarray(tensor.permute(1, 2, 0).cpu().numpy(), 'RGB')


def _pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float().div(255)


# --- 公共退化函数 (已升级) ---

# 1. 深度感知的雨 (Advanced Rain)
def degrade_rain(image: torch.Tensor) -> torch.Tensor:
    bimg = _to_batch(image)
    _, C, H, W = bimg.shape
    d = bimg.device

    output = bimg.clone()
    num_layers = random.randint(2, 4)

    # 创建一个简单的从上到下(远->近)的伪深度图
    pseudo_depth = torch.linspace(0, 1, H, device=d).view(1, 1, H, 1).expand(1, 1, H, W)

    for i in range(num_layers):
        # 深度越深（值越小），雨丝越小、越清晰
        # 深度越浅（值越大），雨丝越大、越模糊
        layer_depth = 1.0 - (i + random.uniform(0.1, 0.9)) / num_layers

        # 参数与深度挂钩
        length = random.randint(int(7 + 15 * layer_depth), int(15 + 25 * layer_depth))
        angle = random.uniform(-20, 20)
        density = random.uniform(0.005, 0.015) * (1 + 0.5 * layer_depth)
        blur_sigma = 0.5 + 1.5 * layer_depth  # 近景更模糊
        intensity = random.uniform(0.15, 0.35)

        seeds = (torch.rand(1, 1, H, W, device=d) < density).float() * (pseudo_depth > (layer_depth - 0.2))

        kernel = _make_motion_kernel(length, angle).to(d)
        streaks = _conv2d_channelwise(seeds, kernel)

        if blur_sigma > 0.5:
            blur_ks = int(blur_sigma * 3) | 1
            blur_kernel = _make_gaussian_kernel(blur_ks, blur_sigma).to(d)
            streaks = _conv2d_channelwise(streaks, blur_kernel)

        streaks = (streaks / streaks.max().clamp_min(1e-6)).clamp(0, 1)
        output = output * (1 - intensity * streaks)

    return _from_batch(output.clamp(0, 1))


# 2. 深度感知的雪 (Advanced Snow)
def degrade_snow(image: torch.Tensor) -> torch.Tensor:
    bimg = _to_batch(image)
    _, C, H, W = bimg.shape
    d = bimg.device

    output = bimg.clone()
    num_layers = random.randint(2, 3)

    for _ in range(num_layers):
        ks = random.choice([7, 9, 11, 13])
        sigma = ks / random.uniform(2.5, 4.0)
        density = random.uniform(0.003, 0.01)
        intensity = random.uniform(0.2, 0.6)

        seeds = (torch.rand(1, 1, H, W, device=d) < density).float()
        kernel = _make_gaussian_kernel(ks, sigma).to(d)
        flakes = _conv2d_channelwise(seeds, kernel)
        flakes = (flakes / flakes.max().clamp_min(1e-6)).clamp(0, 1)

        output = torch.maximum(output, intensity * flakes)

    return _from_batch(output.clamp(0, 1))


# 3. 新增：真实相机噪声
def degrade_camera_noise(image: torch.Tensor) -> torch.Tensor:
    bimg = _to_batch(image)
    shot_noise_strength = random.uniform(0.01, 0.08)
    read_noise_strength = random.uniform(0.005, 0.03)
    shot_noise = torch.randn_like(bimg) * torch.sqrt(bimg.clamp_min(0)) * shot_noise_strength
    read_noise = torch.randn_like(bimg) * read_noise_strength
    noisy_img = bimg + shot_noise + read_noise
    return _from_batch(noisy_img.clamp(0, 1))


# 4. 新增：JPEG 压缩伪影
def degrade_jpeg(image: torch.Tensor) -> torch.Tensor:
    quality = random.randint(20, 75)
    pil_image = _tensor_to_pil(image.cpu())
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    jpeg_image = Image.open(buffer)
    return _pil_to_tensor(jpeg_image).to(image.device)


# --- 保留的其他退化函数 ---
# (此处省略 degrade_cloud, degrade_haze, degrade_fog, degrade_lowlight,
# degrade_blur, degrade_dirty_lens 的代码，假设它们保持原样)
def degrade_cloud(image: torch.Tensor) -> torch.Tensor:
    bimg = _to_batch(image)
    _, C, H, W = bimg.shape
    d = bimg.device
    noise = F.interpolate(torch.rand(1, 1, max(4, H // 24), max(4, W // 24), device=d), size=(H, W), mode="bilinear",
                          align_corners=False).clamp(0, 1)
    y = torch.linspace(0, 1, H, device=d).view(1, 1, H, 1)
    top_mask = (1 - y).pow(1.5)
    mask = (noise * top_mask).clamp(0, 1)
    strength = random.uniform(0.2, 0.6)
    A = torch.ones(1, C, 1, 1, device=d) * random.uniform(0.9, 1.0)
    out = bimg * (1 - strength * mask) + A * (strength * mask)
    return _from_batch(out.clamp(0, 1))



def degrade_fog(image: torch.Tensor) -> torch.Tensor:
    bimg = _to_batch(image)
    _, C, H, W = bimg.shape
    d = bimg.device
    A_val = random.uniform(0.85, 1.0)
    A = torch.ones(1, C, 1, 1, device=d) * A_val
    yy = torch.linspace(-1, 1, H, device=d).view(1, 1, H, 1).abs()
    xx = torch.linspace(-1, 1, W, device=d).view(1, 1, 1, W).abs()
    dist = torch.sqrt(yy ** 2 + xx ** 2)
    height_bias = (torch.linspace(0, 1, H, device=d).view(1, 1, H, 1))
    mask = (1 - dist) * 0.5 + (1 - height_bias) * 0.5
    noise = F.interpolate(torch.rand(1, 1, max(4, H // 20), max(4, W // 20), device=d), size=(H, W), mode="bilinear",
                          align_corners=False).clamp(0, 1)
    fog_mask = (0.6 * mask + 0.4 * noise).clamp(0, 1)
    strength = random.uniform(0.4, 0.9)
    t = (1 - strength * fog_mask).clamp(0.1, 1.0)
    out = bimg * t + A * (1 - t)
    return _from_batch(out.clamp(0, 1))


def degrade_lowlight(image: torch.Tensor) -> torch.Tensor:
    bimg = _to_batch(image)
    d = bimg.device

    # 调整1：收窄 scale 范围，避免过暗
    scale = random.uniform(0.35, 0.65)

    # 调整2：引入少量环境光 (ambient light)，防止纯黑区域
    ambient_light = random.uniform(0.01, 0.05)

    # 应用亮度和环境光
    out = bimg * scale + ambient_light

    # 调整3：Gamma 校正范围也稍微降低，防止对比度过高
    gamma = random.uniform(1.5, 2.5)
    out = out.clamp(0, 1).pow(gamma)

    # 保留相机噪声，因为低光通常伴随噪声
    # 注意：这里的噪声是加在调整亮度之后，更符合物理过程
    out = degrade_camera_noise(out)  # 复用我们已有的真实相机噪声函数

    return _from_batch(out.clamp(0, 1))


def degrade_blur(image: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.5:
        ks = random.choice([3, 5, 7, 9])
        sigma = random.uniform(0.5, ks / 2.0)
        kernel = _make_gaussian_kernel(ks, sigma)
    else:
        length = random.randint(5, 21)
        angle = random.uniform(-90, 90)
        kernel = _make_motion_kernel(length, angle)
    bimg = _to_batch(image)
    out = _conv2d_channelwise(bimg, kernel.to(bimg.device)).clamp(0, 1)
    return _from_batch(out)


def degrade_dirty_lens(image: torch.Tensor) -> torch.Tensor:
    bimg = _to_batch(image)
    _, C, H, W = bimg.shape
    d = bimg.device
    num_spots = random.randint(5, 25)
    yy = torch.linspace(0, 1, H, device=d).view(H, 1).expand(H, W)
    xx = torch.linspace(0, 1, W, device=d).view(1, W).expand(H, W)
    mask = torch.zeros(H, W, device=d)
    for _ in range(num_spots):
        cx, cy = random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)
        r = random.uniform(0.02, 0.12)
        dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        spot = torch.exp(-(dist ** 2) / (2 * (r ** 2)))
        spot *= random.uniform(0.3, 1.0)
        mask = torch.maximum(mask, spot)
    ks = random.choice([9, 11, 13, 15])
    kernel = _make_gaussian_kernel(ks, ks / 3.0).to(d)
    mask_b = mask.view(1, 1, H, W)
    mask_b = _conv2d_channelwise(mask_b.repeat(1, C, 1, 1), kernel).mean(dim=1, keepdim=True)
    mask_b = (mask_b / (mask_b.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6))).clamp(0, 1)
    tint_color = torch.tensor([0.85, 0.8, 0.75], device=d).view(1, 3, 1, 1)
    alpha = random.uniform(0.2, 0.6)
    out = bimg * (1 - alpha * mask_b) + tint_color * (alpha * mask_b)
    if random.random() < 0.5:
        yy = torch.linspace(-1, 1, H, device=d).view(H, 1).expand(H, W)
        xx = torch.linspace(-1, 1, W, device=d).view(1, W).expand(H, W)
        vign = 1 - 0.3 * (xx ** 2 + yy ** 2)
        vign = vign.clamp(0.5, 1.0).view(1, 1, H, W)
        out = (out * vign).clamp(0, 1)
    return _from_batch(out.clamp(0, 1))


# --- 5. 复合退化管道 ---

ALL_DEGRADATIONS = {
    "rain": degrade_rain,
    "snow": degrade_snow,
    "blur": degrade_blur,
    "camera_noise": degrade_camera_noise,
    "jpeg": degrade_jpeg,
    "lowlight": degrade_lowlight,
    "dirty_lens": degrade_dirty_lens,
    "fog": degrade_fog,
    "cloud": degrade_cloud,
}

# 定义互斥规则
# 每个集合内的元素是互斥的，即一次复合退化中最多只能出现一个。
EXCLUSION_GROUPS = [
    {'rain', 'snow', 'fog', 'cloud'},  # 天气条件互斥
    {'lowlight', 'cloud', 'fog'}  # 光照条件与部分天气互斥
]


def apply_mixed_degradations(image: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    随机选择 1 到 k 种退化，并按随机顺序应用。
    此版本会检查互斥规则，以避免生成不合逻辑的场景。
    """
    num_to_apply = random.randint(1, k)

    # 可用的退化选项池
    available_degradations = set(ALL_DEGRADATIONS.keys())
    chosen_degradations = []

    # 循环选择退化，直到达到数量要求或无可用选项
    for _ in range(num_to_apply):
        if not available_degradations:
            break

        # 1. 从当前可用的池中随机选择一个
        choice = random.choice(list(available_degradations))
        chosen_degradations.append(choice)

        # 2. 从可用池中移除刚刚选择的那个
        available_degradations.remove(choice)

        # 3. 核心逻辑：检查所有互斥规则
        # 如果选中的退化属于任何一个互斥组，就从可用池中移除该组的其他所有成员
        for group in EXCLUSION_GROUPS:
            if choice in group:
                # 使用集合的差集操作来移除所有冲突项
                conflicts = group - {choice}
                available_degradations -= conflicts

    degraded_image = image.clone()
    # 应用顺序仍然随机化，以增加多样性
    random.shuffle(chosen_degradations)

    # (可选) 增加一个打印语句，方便您在测试时查看选择了哪些组合
    print(f"应用场景: {chosen_degradations}")

    for name in chosen_degradations:
        degraded_image = ALL_DEGRADATIONS[name](degraded_image)

    return degraded_image.clamp(0, 1)


