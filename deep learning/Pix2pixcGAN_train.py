#%%
import csv
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from monai.utils import set_determinism
from pytorch_msssim import ssim

from dataset import VirtualStainingPatchDataset
from utils.training import RunConfig, clip_gradients


# =========================================================
# 1. Config
# =========================================================
@dataclass
class Config(RunConfig):
    # -------- data --------
    data_root: str = str(Path(__file__).resolve().parent / "dataset_patches")
    task: str = "BFtoIBA1"   # BFtoDAPI / BFtoCD206 / BFtoLYVE1 / BFtoNeutrophil
    in_channels: int = 3
    out_channels: int = 1
    image_size: int = 256
    base_channels: int = 64

    # -------- training --------
    epochs: int = 300
    batch_size: int = 4
    num_workers: int = 0
    lr_G: float = 2e-4
    lr_D: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_l1: float = 100.0
    seed: int = 42
    amp: bool = True

    # -------- visualization --------
    save_sample_every: int = 10
    num_sample_images: int = 4

    # -------- metrics --------
    data_range: float = 1.0
    clamp_for_metrics: bool = True

    # -------- output --------
    output_root: str = str(Path(__file__).resolve().parent / "runs")


# =========================================================
# 2. Utils
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int):
    set_determinism(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_batch_tensors(batch, device: torch.device):
    """
    兼容两种 Dataset 返回:
    1) dict: {"input":..., "target":...}
    2) tuple: (input, target)
    """
    if isinstance(batch, dict):
        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
    else:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
    return x, y


from utils.metrics import (
    prepare_for_metrics, compute_mae, compute_nmse, compute_psnr,
    compute_ssim, compute_all_metrics,
)
from utils.training import parse_config, validate_config, create_run_dir, check_datasets


def save_sample_images(generator, loader, device, save_path: Path, max_samples: int = 4):
    """
    保存可视化:
    BF | GT | Pred
    对 BFtoIBA1:
      GT / Pred 使用红色风格显示
    """
    generator.eval()

    batch = next(iter(loader))
    x, y = get_batch_tensors(batch, device)

    with torch.no_grad():
        pred = generator(x)

    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()

    n = min(max_samples, x.shape[0])

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        # 1. Input BF
        img = x[i]  # C,H,W
        if img.shape[0] == 3:
            bf = np.transpose(img, (1, 2, 0))
        else:
            bf = img[0]
        bf = np.clip(bf, 0, 1)

        axes[i, 0].imshow(bf, cmap=None if bf.ndim == 3 else "gray")
        axes[i, 0].set_title("BF")
        axes[i, 0].axis("off")

        # 2. GT
        gt = np.clip(y[i, 0], 0, 1)
        gt_rgb = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.float32)
        gt_rgb[..., 0] = gt
        axes[i, 1].imshow(gt_rgb)
        axes[i, 1].set_title("GT")
        axes[i, 1].axis("off")

        # 3. Pred
        pr = np.clip(pred[i, 0], 0, 1)
        pr_rgb = np.zeros((pr.shape[0], pr.shape[1], 3), dtype=np.float32)
        pr_rgb[..., 0] = pr
        axes[i, 2].imshow(pr_rgb)
        axes[i, 2].set_title("Pred")
        axes[i, 2].axis("off")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def init_weights(net):
    """
    pix2pix 常见初始化方式
    """
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)


# =========================================================
# 3. Model: U-Net Generator
# =========================================================
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), dim=1)
        return x


class GeneratorUNet(nn.Module):
    """
    标准 pix2pix 风格 U-Net generator
    输入 256x256 时比较合适
    """
    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super().__init__()
        c = base_channels

        self.down1 = UNetDown(in_channels, c, normalize=False)      # 256 -> 128
        self.down2 = UNetDown(c, 2*c)                               # 128 -> 64
        self.down3 = UNetDown(2*c, 4*c)                              # 64 -> 32
        self.down4 = UNetDown(4*c, 8*c, dropout=0.0)                # 32 -> 16
        self.down5 = UNetDown(8*c, 8*c, dropout=0.0)                # 16 -> 8
        self.down6 = UNetDown(8*c, 8*c, dropout=0.0)                # 8 -> 4
        self.down7 = UNetDown(8*c, 8*c, dropout=0.0)                # 4 -> 2
        self.down8 = UNetDown(8*c, 8*c, normalize=False, dropout=0.0)  # 2 -> 1

        self.up1 = UNetUp(8*c, 8*c, dropout=0.5)                    # 1 -> 2
        self.up2 = UNetUp(16*c, 8*c, dropout=0.5)                   # 2 -> 4
        self.up3 = UNetUp(16*c, 8*c, dropout=0.5)                   # 4 -> 8
        self.up4 = UNetUp(16*c, 8*c, dropout=0.0)                   # 8 -> 16
        self.up5 = UNetUp(16*c, 4*c, dropout=0.0)                   # 16 -> 32
        self.up6 = UNetUp(8*c, 2*c, dropout=0.0)                    # 32 -> 64
        self.up7 = UNetUp(4*c, c, dropout=0.0)                     # 64 -> 128

        self.final = nn.Sequential(
            nn.ConvTranspose2d(2*c, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()   # 你的图像如果是[0,1]，这里用 Sigmoid 更合适
        )

    def forward(self, x):
        if x.shape[-2] % 256 or x.shape[-1] % 256:
            raise ValueError("Pix2Pix requires spatial dimensions divisible by 256")
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


# =========================================================
# 4. Model: PatchGAN Discriminator
# =========================================================
class Discriminator(nn.Module):
    """
    PatchGAN discriminator
    输入为 concat(input, target) 或 concat(input, fake)
    """
    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super().__init__()
        c = base_channels

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [
                nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)
            ]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels + out_channels, c, normalize=False),  # 256 -> 128
            *discriminator_block(c, 2*c),                                           # 128 -> 64
            *discriminator_block(2*c, 4*c),                                          # 64 -> 32
            *discriminator_block(4*c, 8*c),                                          # 32 -> 16
            nn.Conv2d(8*c, 1, kernel_size=4, stride=1, padding=1)                    # Patch output
        )

    def forward(self, img_A, img_B):
        # img_A: input
        # img_B: target or generated
        x = torch.cat((img_A, img_B), dim=1)
        return self.model(x)


# =========================================================
# 5. Loss
# =========================================================
class Pix2PixLoss(nn.Module):
    def __init__(self, lambda_l1=100.0):
        super().__init__()
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()
        self.lambda_l1 = lambda_l1

    def generator_loss(self, pred_fake, fake_img, real_img):
        valid = torch.ones_like(pred_fake, device=pred_fake.device)
        loss_gan = self.criterion_gan(pred_fake, valid)
        loss_l1 = self.criterion_l1(fake_img, real_img)
        loss_g = loss_gan + self.lambda_l1 * loss_l1
        return loss_g, loss_gan, loss_l1

    def discriminator_loss(self, pred_real, pred_fake):
        valid = torch.ones_like(pred_real, device=pred_real.device)
        fake = torch.zeros_like(pred_fake, device=pred_fake.device)

        loss_real = self.criterion_gan(pred_real, valid)
        loss_fake = self.criterion_gan(pred_fake, fake)
        loss_d = 0.5 * (loss_real + loss_fake)
        return loss_d, loss_real, loss_fake


# =========================================================
# 6. Train / Val
# =========================================================
def train_one_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    loader: DataLoader,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    criterion: Pix2PixLoss,
    device: torch.device,
    scaler_G: GradScaler,
    scaler_D: GradScaler,
    data_range: float = 1.0,
    clamp_for_metrics: bool = True,
    use_amp: bool = True,
    grad_clip: float = 1.0,
):
    generator.train()
    discriminator.train()

    total_loss_G = 0.0
    total_loss_D = 0.0
    total_loss_GAN = 0.0
    total_loss_L1 = 0.0

    total_mae = 0.0
    total_nmse = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    n = 0

    for batch in loader:
        real_A, real_B = get_batch_tensors(batch, device)

        # -------------------------
        # 1) Train Generator
        # -------------------------
        discriminator.requires_grad_(False)
        optimizer_G.zero_grad(set_to_none=True)

        with autocast(device.type, enabled=use_amp and device.type == "cuda"):
            fake_B = generator(real_A)
            pred_fake = discriminator(real_A, fake_B)
            loss_G, loss_GAN, loss_L1 = criterion.generator_loss(pred_fake, fake_B, real_B)

        scaler_G.scale(loss_G).backward()
        clip_gradients(scaler_G, optimizer_G, grad_clip)
        scaler_G.step(optimizer_G)
        scaler_G.update()

        # -------------------------
        # 2) Train Discriminator
        # -------------------------
        discriminator.requires_grad_(True)
        optimizer_D.zero_grad(set_to_none=True)

        with autocast(device.type, enabled=use_amp and device.type == "cuda"):
            pred_real = discriminator(real_A, real_B)
            pred_fake_detached = discriminator(real_A, fake_B.detach())
            loss_D, loss_real, loss_fake = criterion.discriminator_loss(pred_real, pred_fake_detached)

        scaler_D.scale(loss_D).backward()
        clip_gradients(scaler_D, optimizer_D, grad_clip)
        scaler_D.step(optimizer_D)
        scaler_D.update()

        # -------------------------
        # metrics
        # -------------------------
        metrics = compute_all_metrics(
            pred=fake_B.detach(),
            target=real_B,
            data_range=data_range,
            clamp_for_metrics=clamp_for_metrics,
        )

        bs = real_A.size(0)
        total_loss_G += loss_G.item() * bs
        total_loss_D += loss_D.item() * bs
        total_loss_GAN += loss_GAN.item() * bs
        total_loss_L1 += loss_L1.item() * bs

        total_mae += metrics["mae"] * bs
        total_nmse += metrics["nmse"] * bs
        total_ssim += metrics["ssim"] * bs
        total_psnr += metrics["psnr"] * bs
        n += bs

    if n == 0:
        raise ValueError("DataLoader produced no batches")

    return {
        "loss_G": total_loss_G / max(n, 1),
        "loss_D": total_loss_D / max(n, 1),
        "loss_GAN": total_loss_GAN / max(n, 1),
        "loss_L1": total_loss_L1 / max(n, 1),
        "mae": total_mae / max(n, 1),
        "nmse": total_nmse / max(n, 1),
        "ssim": total_ssim / max(n, 1),
        "psnr": total_psnr / max(n, 1),
    }


@torch.no_grad()
def validate_one_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    loader: DataLoader,
    criterion: Pix2PixLoss,
    device: torch.device,
    data_range: float = 1.0,
    clamp_for_metrics: bool = True,
    use_amp: bool = True,
):
    generator.eval()
    discriminator.eval()

    total_loss_G = 0.0
    total_loss_D = 0.0
    total_loss_GAN = 0.0
    total_loss_L1 = 0.0

    total_mae = 0.0
    total_nmse = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    n = 0

    for batch in loader:
        real_A, real_B = get_batch_tensors(batch, device)

        with autocast(device.type, enabled=use_amp and device.type == "cuda"):
            fake_B = generator(real_A)
            pred_fake = discriminator(real_A, fake_B)
            pred_real = discriminator(real_A, real_B)

            loss_G, loss_GAN, loss_L1 = criterion.generator_loss(pred_fake, fake_B, real_B)
            loss_D, _, _ = criterion.discriminator_loss(pred_real, pred_fake)

        metrics = compute_all_metrics(
            pred=fake_B,
            target=real_B,
            data_range=data_range,
            clamp_for_metrics=clamp_for_metrics,
        )

        bs = real_A.size(0)
        total_loss_G += loss_G.item() * bs
        total_loss_D += loss_D.item() * bs
        total_loss_GAN += loss_GAN.item() * bs
        total_loss_L1 += loss_L1.item() * bs

        total_mae += metrics["mae"] * bs
        total_nmse += metrics["nmse"] * bs
        total_ssim += metrics["ssim"] * bs
        total_psnr += metrics["psnr"] * bs
        n += bs

    if n == 0:
        raise ValueError("DataLoader produced no batches")

    return {
        "loss_G": total_loss_G / max(n, 1),
        "loss_D": total_loss_D / max(n, 1),
        "loss_GAN": total_loss_GAN / max(n, 1),
        "loss_L1": total_loss_L1 / max(n, 1),
        "mae": total_mae / max(n, 1),
        "nmse": total_nmse / max(n, 1),
        "ssim": total_ssim / max(n, 1),
        "psnr": total_psnr / max(n, 1),
    }


# =========================================================
# 7. Run
# =========================================================
def main(cfg=None):
    import sys
    from runner import run_training
    return run_training('pix2pix', sys.modules[__name__], cfg or Config())


if __name__ == '__main__':
    main(parse_config(Config))
