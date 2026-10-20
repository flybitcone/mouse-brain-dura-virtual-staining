#%%
import csv
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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
    in_channels_A: int = 3   # BF
    in_channels_B: int = 1   # stain
    image_size: int = 256
    base_channels: int = 64
    n_residual_blocks: int = 9

    # -------- training --------
    epochs: int = 300
    batch_size: int = 2
    num_workers: int = 0
    lr_G: float = 2e-4
    lr_D: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999

    lambda_cycle: float = 10.0
    lambda_identity: float = 0.0  # Identity loss is undefined for 3-channel <-> 1-channel domains

    seed: int = 42
    amp: bool = True

    # -------- visualization --------
    save_sample_every: int = 10
    num_sample_images: int = 4

    # -------- metrics --------
    data_range: float = 1.0
    clamp_for_metrics: bool = True

    # -------- image pool --------
    pool_size: int = 50

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
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


from utils.metrics import (
    prepare_for_metrics, compute_mae, compute_nmse, compute_psnr,
    compute_ssim, compute_all_metrics,
)
from utils.training import parse_config, validate_config, create_run_dir, check_datasets


def init_weights(net):
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


def tensor_to_rgb_bf(x):
    # x: [C,H,W]
    if x.shape[0] == 3:
        img = np.transpose(x, (1, 2, 0))
        img = np.clip(img, 0, 1)
        return img
    else:
        img = np.clip(x[0], 0, 1)
        return img


def tensor_to_red_rgb(x2d):
    # x2d: [H,W]
    x2d = np.clip(x2d, 0, 1)
    rgb = np.zeros((x2d.shape[0], x2d.shape[1], 3), dtype=np.float32)
    rgb[..., 0] = x2d
    return rgb


# =========================================================
# 3. Dataset wrapper
# =========================================================
class UnpairedVirtualStainingDataset(Dataset):
    """
    兼容你现在的 paired dataset:
      原始 dataset 返回 {"input": BF, "target": stain}
    这里把它拆成 A 域和 B 域
    训练时:
      A 取当前 index 的 input
      B 随机取另一个样本的 target
    验证/测试时:
      A 和 B 使用同一样本中的 input/target，便于算指标
    """
    def __init__(self, root, task, subset="train", return_metadata=False, unpaired=True):
        self.base_ds = VirtualStainingPatchDataset(
            root=root,
            task=task,
            subset=subset,
            return_metadata=return_metadata,
        )
        self.unpaired = unpaired
        self.subset = subset

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, index):
        sample_A = self.base_ds[index]

        if isinstance(sample_A, dict):
            real_A = sample_A["input"]
            real_B_pair = sample_A["target"]
        else:
            real_A, real_B_pair = sample_A

        if self.unpaired and self.subset == "train":
            rand_idx = random.randint(0, len(self.base_ds) - 1)
            sample_B = self.base_ds[rand_idx]
            if isinstance(sample_B, dict):
                real_B = sample_B["target"]
            else:
                _, real_B = sample_B
        else:
            real_B = real_B_pair

        return {
            "A": real_A,         # BF
            "B": real_B,         # stain (possibly unpaired in train)
            "B_pair": real_B_pair,  # paired GT only for metric/vis if needed
        }


def get_batch_tensors(batch, device: torch.device):
    real_A = batch["A"].to(device, non_blocking=True)
    real_B = batch["B"].to(device, non_blocking=True)
    real_B_pair = batch["B_pair"].to(device, non_blocking=True)
    return real_A, real_B, real_B_pair


# =========================================================
# 4. Model: ResNet Generator (CycleGAN standard)
# =========================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, n_residual_blocks=9, base_channels=64):
        super().__init__()
        c = base_channels

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, c, kernel_size=7, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(c),
            nn.ReLU(inplace=True),
        ]

        curr_dim = c

        # Downsample
        for _ in range(2):
            model += [
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2),
                nn.ReLU(inplace=True),
            ]
            curr_dim *= 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(curr_dim)]

        # Upsample
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(
                    curr_dim, curr_dim // 2,
                    kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
                ),
                nn.InstanceNorm2d(curr_dim // 2),
                nn.ReLU(inplace=True),
            ]
            curr_dim = curr_dim // 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(curr_dim, out_channels, kernel_size=7, stride=1, padding=0),
            nn.Sigmoid(),   # 因为你的图像是 [0,1]
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# =========================================================
# 5. Model: PatchGAN Discriminator
# =========================================================
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        c = base_channels

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, c, normalize=False),
            *block(c, 2*c, normalize=True),
            *block(2*c, 4*c, normalize=True),
            *block(4*c, 8*c, normalize=True),
            nn.Conv2d(8*c, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)


# =========================================================
# 6. Image Pool
# =========================================================
class ImagePool:
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = deque(maxlen=pool_size)

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.pool_size == 0:
            return images

        return_images = []
        for image in images:
            image = image.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(image.detach().clone())
                return_images.append(image)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.pool_size - 1)
                    tmp = self.images[idx].clone()
                    self.images[idx] = image.detach().clone()
                    return_images.append(tmp)
                else:
                    return_images.append(image)

        return torch.cat(return_images, dim=0)


# =========================================================
# 7. Loss
# =========================================================
class CycleGANLoss(nn.Module):
    def __init__(self, lambda_cycle=10.0, lambda_identity=0.0):
        super().__init__()
        self.criterion_gan = nn.MSELoss()   # LSGAN
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def generator_loss(
        self,
        real_A, real_B,
        fake_A, fake_B,
        rec_A, rec_B,
        idt_A, idt_B,
        pred_fake_A, pred_fake_B
    ):
        valid_A = torch.ones_like(pred_fake_A, device=pred_fake_A.device)
        valid_B = torch.ones_like(pred_fake_B, device=pred_fake_B.device)

        loss_gan_A2B = self.criterion_gan(pred_fake_B, valid_B)
        loss_gan_B2A = self.criterion_gan(pred_fake_A, valid_A)

        loss_cycle_A = self.criterion_cycle(rec_A, real_A)
        loss_cycle_B = self.criterion_cycle(rec_B, real_B)

        loss_idt_A = real_A.new_zeros(())
        loss_idt_B = real_B.new_zeros(())
        if self.lambda_identity:
            if idt_A is None or idt_B is None:
                raise ValueError("Identity predictions are required when identity loss is enabled")
            loss_idt_A = self.criterion_identity(idt_A, real_A)
            loss_idt_B = self.criterion_identity(idt_B, real_B)

        loss_cycle = loss_cycle_A + loss_cycle_B
        loss_identity = loss_idt_A + loss_idt_B
        loss_gan = loss_gan_A2B + loss_gan_B2A

        loss_G = (
            loss_gan
            + self.lambda_cycle * loss_cycle
            + self.lambda_identity * loss_identity
        )

        return {
            "loss_G": loss_G,
            "loss_gan": loss_gan,
            "loss_gan_A2B": loss_gan_A2B,
            "loss_gan_B2A": loss_gan_B2A,
            "loss_cycle": loss_cycle,
            "loss_cycle_A": loss_cycle_A,
            "loss_cycle_B": loss_cycle_B,
            "loss_identity": loss_identity,
            "loss_idt_A": loss_idt_A,
            "loss_idt_B": loss_idt_B,
        }

    def discriminator_loss(self, pred_real, pred_fake):
        valid = torch.ones_like(pred_real, device=pred_real.device)
        fake = torch.zeros_like(pred_fake, device=pred_fake.device)

        loss_real = self.criterion_gan(pred_real, valid)
        loss_fake = self.criterion_gan(pred_fake, fake)
        loss_D = 0.5 * (loss_real + loss_fake)
        return loss_D, loss_real, loss_fake


# =========================================================
# 8. Visualization
# =========================================================
@torch.no_grad()
def save_sample_images(
    G_AB,
    G_BA,
    loader,
    device,
    save_path: Path,
    max_samples: int = 4,
):
    G_AB.eval()
    G_BA.eval()

    batch = next(iter(loader))
    real_A, real_B, real_B_pair = get_batch_tensors(batch, device)

    fake_B = G_AB(real_A)
    rec_A = G_BA(fake_B)

    fake_A = G_BA(real_B_pair)
    rec_B = G_AB(fake_A)

    real_A = real_A.detach().cpu().numpy()
    real_B_pair = real_B_pair.detach().cpu().numpy()
    fake_B = fake_B.detach().cpu().numpy()
    rec_A = rec_A.detach().cpu().numpy()

    n = min(max_samples, real_A.shape[0])

    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        bf = tensor_to_rgb_bf(real_A[i])
        gt = tensor_to_red_rgb(real_B_pair[i, 0])
        pred = tensor_to_red_rgb(fake_B[i, 0])
        cycle_a = tensor_to_rgb_bf(rec_A[i])

        axes[i, 0].imshow(bf, cmap=None if bf.ndim == 3 else "gray")
        axes[i, 0].set_title("Real A (BF)")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt)
        axes[i, 1].set_title("Paired GT B")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred)
        axes[i, 2].set_title("Fake B")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(cycle_a, cmap=None if cycle_a.ndim == 3 else "gray")
        axes[i, 3].set_title("Rec A")
        axes[i, 3].axis("off")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# =========================================================
# 9. Train / Val
# =========================================================
def train_one_epoch(
    G_AB: nn.Module,
    G_BA: nn.Module,
    D_A: nn.Module,
    D_B: nn.Module,
    loader: DataLoader,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D_A: torch.optim.Optimizer,
    optimizer_D_B: torch.optim.Optimizer,
    criterion: CycleGANLoss,
    fake_A_pool: ImagePool,
    fake_B_pool: ImagePool,
    device: torch.device,
    scaler_G: GradScaler,
    scaler_D_A: GradScaler,
    scaler_D_B: GradScaler,
    data_range: float = 1.0,
    clamp_for_metrics: bool = True,
    use_amp: bool = True,
    grad_clip: float = 1.0,
):
    G_AB.train()
    G_BA.train()
    D_A.train()
    D_B.train()

    total_loss_G = 0.0
    total_loss_D_A = 0.0
    total_loss_D_B = 0.0
    total_loss_gan = 0.0
    total_loss_cycle = 0.0
    total_loss_identity = 0.0

    # 用 paired GT(B_pair) 评估 A->B 的翻译质量
    total_mae = 0.0
    total_nmse = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    n = 0

    for batch in loader:
        real_A, real_B, real_B_pair = get_batch_tensors(batch, device)

        # -------------------------
        # 1) Train Generators
        # -------------------------
        D_A.requires_grad_(False)
        D_B.requires_grad_(False)
        optimizer_G.zero_grad(set_to_none=True)

        with autocast(device.type, enabled=use_amp and device.type == "cuda"):
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)

            rec_A = G_BA(fake_B)
            rec_B = G_AB(fake_A)

            idt_A = idt_B = None
            if criterion.lambda_identity:
                if real_A.shape[1] != real_B.shape[1]:
                    raise ValueError("Identity loss requires equal domain channels")
                idt_A = G_BA(real_A)
                idt_B = G_AB(real_B)

            pred_fake_A = D_A(fake_A)
            pred_fake_B = D_B(fake_B)

            g_losses = criterion.generator_loss(
                real_A=real_A,
                real_B=real_B,
                fake_A=fake_A,
                fake_B=fake_B,
                rec_A=rec_A,
                rec_B=rec_B,
                idt_A=idt_A,
                idt_B=idt_B,
                pred_fake_A=pred_fake_A,
                pred_fake_B=pred_fake_B,
            )

            loss_G = g_losses["loss_G"]

        scaler_G.scale(loss_G).backward()
        clip_gradients(scaler_G, optimizer_G, grad_clip)
        scaler_G.step(optimizer_G)
        scaler_G.update()

        # -------------------------
        # 2) Train D_A
        # -------------------------
        D_A.requires_grad_(True)
        D_B.requires_grad_(True)
        optimizer_D_A.zero_grad(set_to_none=True)

        with autocast(device.type, enabled=use_amp and device.type == "cuda"):
            fake_A_buffer = fake_A_pool.query(fake_A.detach())
            pred_real_A = D_A(real_A)
            pred_fake_A = D_A(fake_A_buffer)
            loss_D_A, _, _ = criterion.discriminator_loss(pred_real_A, pred_fake_A)

        scaler_D_A.scale(loss_D_A).backward()
        clip_gradients(scaler_D_A, optimizer_D_A, grad_clip)
        scaler_D_A.step(optimizer_D_A)
        scaler_D_A.update()

        # -------------------------
        # 3) Train D_B
        # -------------------------
        optimizer_D_B.zero_grad(set_to_none=True)

        with autocast(device.type, enabled=use_amp and device.type == "cuda"):
            fake_B_buffer = fake_B_pool.query(fake_B.detach())
            pred_real_B = D_B(real_B)
            pred_fake_B = D_B(fake_B_buffer)
            loss_D_B, _, _ = criterion.discriminator_loss(pred_real_B, pred_fake_B)

        scaler_D_B.scale(loss_D_B).backward()
        clip_gradients(scaler_D_B, optimizer_D_B, grad_clip)
        scaler_D_B.step(optimizer_D_B)
        scaler_D_B.update()

        # -------------------------
        # metrics (A->B vs paired GT)
        # -------------------------
        metrics = compute_all_metrics(
            pred=fake_B.detach(),
            target=real_B_pair,
            data_range=data_range,
            clamp_for_metrics=clamp_for_metrics,
        )

        bs = real_A.size(0)
        total_loss_G += g_losses["loss_G"].item() * bs
        total_loss_D_A += loss_D_A.item() * bs
        total_loss_D_B += loss_D_B.item() * bs
        total_loss_gan += g_losses["loss_gan"].item() * bs
        total_loss_cycle += g_losses["loss_cycle"].item() * bs
        total_loss_identity += g_losses["loss_identity"].item() * bs

        total_mae += metrics["mae"] * bs
        total_nmse += metrics["nmse"] * bs
        total_ssim += metrics["ssim"] * bs
        total_psnr += metrics["psnr"] * bs
        n += bs

    if n == 0:
        raise ValueError("DataLoader produced no batches")

    return {
        "loss_G": total_loss_G / max(n, 1),
        "loss_D_A": total_loss_D_A / max(n, 1),
        "loss_D_B": total_loss_D_B / max(n, 1),
        "loss_gan": total_loss_gan / max(n, 1),
        "loss_cycle": total_loss_cycle / max(n, 1),
        "loss_identity": total_loss_identity / max(n, 1),
        "mae": total_mae / max(n, 1),
        "nmse": total_nmse / max(n, 1),
        "ssim": total_ssim / max(n, 1),
        "psnr": total_psnr / max(n, 1),
    }


@torch.no_grad()
def validate_one_epoch(
    G_AB: nn.Module,
    G_BA: nn.Module,
    D_A: nn.Module,
    D_B: nn.Module,
    loader: DataLoader,
    criterion: CycleGANLoss,
    device: torch.device,
    data_range: float = 1.0,
    clamp_for_metrics: bool = True,
    use_amp: bool = True,
):
    G_AB.eval()
    G_BA.eval()
    D_A.eval()
    D_B.eval()

    total_loss_G = 0.0
    total_loss_D_A = 0.0
    total_loss_D_B = 0.0
    total_loss_gan = 0.0
    total_loss_cycle = 0.0
    total_loss_identity = 0.0

    total_mae = 0.0
    total_nmse = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    n = 0

    for batch in loader:
        real_A, real_B, real_B_pair = get_batch_tensors(batch, device)

        with autocast(device.type, enabled=use_amp and device.type == "cuda"):
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)

            rec_A = G_BA(fake_B)
            rec_B = G_AB(fake_A)

            idt_A = idt_B = None
            if criterion.lambda_identity:
                if real_A.shape[1] != real_B.shape[1]:
                    raise ValueError("Identity loss requires equal domain channels")
                idt_A = G_BA(real_A)
                idt_B = G_AB(real_B)

            pred_fake_A = D_A(fake_A)
            pred_fake_B = D_B(fake_B)

            g_losses = criterion.generator_loss(
                real_A=real_A,
                real_B=real_B,
                fake_A=fake_A,
                fake_B=fake_B,
                rec_A=rec_A,
                rec_B=rec_B,
                idt_A=idt_A,
                idt_B=idt_B,
                pred_fake_A=pred_fake_A,
                pred_fake_B=pred_fake_B,
            )

            pred_real_A = D_A(real_A)
            pred_real_B = D_B(real_B)

            loss_D_A, _, _ = criterion.discriminator_loss(pred_real_A, pred_fake_A)
            loss_D_B, _, _ = criterion.discriminator_loss(pred_real_B, pred_fake_B)

        metrics = compute_all_metrics(
            pred=fake_B,
            target=real_B_pair,
            data_range=data_range,
            clamp_for_metrics=clamp_for_metrics,
        )

        bs = real_A.size(0)
        total_loss_G += g_losses["loss_G"].item() * bs
        total_loss_D_A += loss_D_A.item() * bs
        total_loss_D_B += loss_D_B.item() * bs
        total_loss_gan += g_losses["loss_gan"].item() * bs
        total_loss_cycle += g_losses["loss_cycle"].item() * bs
        total_loss_identity += g_losses["loss_identity"].item() * bs

        total_mae += metrics["mae"] * bs
        total_nmse += metrics["nmse"] * bs
        total_ssim += metrics["ssim"] * bs
        total_psnr += metrics["psnr"] * bs
        n += bs

    if n == 0:
        raise ValueError("DataLoader produced no batches")

    return {
        "loss_G": total_loss_G / max(n, 1),
        "loss_D_A": total_loss_D_A / max(n, 1),
        "loss_D_B": total_loss_D_B / max(n, 1),
        "loss_gan": total_loss_gan / max(n, 1),
        "loss_cycle": total_loss_cycle / max(n, 1),
        "loss_identity": total_loss_identity / max(n, 1),
        "mae": total_mae / max(n, 1),
        "nmse": total_nmse / max(n, 1),
        "ssim": total_ssim / max(n, 1),
        "psnr": total_psnr / max(n, 1),
    }


# =========================================================
# 10. Run
# =========================================================
def main(cfg=None):
    import sys
    from runner import run_training
    return run_training('cyclegan', sys.modules[__name__], cfg or Config())


if __name__ == '__main__':
    main(parse_config(Config))
