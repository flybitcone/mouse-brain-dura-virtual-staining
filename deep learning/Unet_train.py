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

from monai.networks.nets import BasicUNet
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

    # -------- training --------
    epochs: int = 300
    batch_size: int = 8
    num_workers: int = 0
    lr: float = 2e-4
    weight_decay: float = 1e-4
    seed: int = 42
    amp: bool = True

    # -------- model --------
    features: tuple = (32, 32, 64, 128, 256, 32)

    # -------- visualization --------
    save_sample_every: int = 10
    num_sample_images: int = 4

    # -------- metrics --------
    data_range: float = 1.0   # 如果 GT 是 [0,1]，保持 1.0；如果是 [0,255] 改成 255.0
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


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


def save_sample_images(model, loader, device, save_path: Path, max_samples: int = 4):
    """
    保存可视化:
    BF | GT | Pred

    对 BFtoIBA1:
      GT / Pred 使用红色风格显示
    """
    model.eval()

    batch = next(iter(loader))
    x, y = get_batch_tensors(batch, device)

    with torch.no_grad():
        pred = model(x)

    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()

    n = min(max_samples, x.shape[0])

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        # -------------------------
        # 1. Input BF
        # -------------------------
        img = x[i]  # C,H,W

        if img.shape[0] == 3:
            bf = np.transpose(img, (1, 2, 0))
        else:
            bf = img[0]

        bf = np.clip(bf, 0, 1)

        axes[i, 0].imshow(bf, cmap=None if bf.ndim == 3 else "gray")
        axes[i, 0].set_title("BF")
        axes[i, 0].axis("off")

        # -------------------------
        # 2. GT
        # -------------------------
        gt = np.clip(y[i, 0], 0, 1)
        gt_rgb = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.float32)
        gt_rgb[..., 0] = gt

        axes[i, 1].imshow(gt_rgb)
        axes[i, 1].set_title("GT")
        axes[i, 1].axis("off")

        # -------------------------
        # 3. Prediction
        # -------------------------
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


# =========================================================
# 3. Model
# =========================================================
def build_model(cfg: Config) -> nn.Module:
    """
    MONAI BasicUNet, 2D version

    upsample="nontrainable"
    使用插值上采样，通常比转置卷积更不容易出现棋盘状伪影
    """
    model = BasicUNet(
        spatial_dims=2,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        features=cfg.features,
        act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm=("INSTANCE", {"affine": True}),
        bias=True,
        dropout=0.0,
        upsample="nontrainable",
    )
    return model


# =========================================================
# 4. Loss
# =========================================================
class CombinedLoss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_ssim=0.2):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim

    def forward(self, pred, target):
        if pred.shape != target.shape:
            raise ValueError("Prediction/target shapes differ")
        with autocast(pred.device.type, enabled=False):
            pred, target = pred.float(), target.float()
            l1 = self.l1(pred, target)
            ssim_loss = 1 - ssim(pred, target, data_range=1.0, size_average=True)
            return self.lambda_l1 * l1 + self.lambda_ssim * ssim_loss


# =========================================================
# 5. Train / Val
# =========================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    data_range: float = 1.0,
    clamp_for_metrics: bool = True,
    use_amp: bool = True,
    grad_clip: float = 1.0,
):
    model.train()

    total_loss = 0.0
    total_mae = 0.0
    total_nmse = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    n = 0

    for batch in loader:
        x, y = get_batch_tensors(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device.type, enabled=use_amp and device.type == "cuda"):
            pred = model(x)
            loss = criterion(pred, y)

        scaler.scale(loss).backward()
        clip_gradients(scaler, optimizer, grad_clip)
        scaler.step(optimizer)
        scaler.update()

        metrics = compute_all_metrics(
            pred=pred.detach(),
            target=y,
            data_range=data_range,
            clamp_for_metrics=clamp_for_metrics,
        )

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_mae += metrics["mae"] * bs
        total_nmse += metrics["nmse"] * bs
        total_ssim += metrics["ssim"] * bs
        total_psnr += metrics["psnr"] * bs
        n += bs

    if n == 0:
        raise ValueError("DataLoader produced no batches")

    return {
        "loss": total_loss / max(n, 1),
        "mae": total_mae / max(n, 1),
        "nmse": total_nmse / max(n, 1),
        "ssim": total_ssim / max(n, 1),
        "psnr": total_psnr / max(n, 1),
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    data_range: float = 1.0,
    clamp_for_metrics: bool = True,
    use_amp: bool = True,
):
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_nmse = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    n = 0

    for batch in loader:
        x, y = get_batch_tensors(batch, device)

        with autocast(device.type, enabled=use_amp and device.type == "cuda"):
            pred = model(x)
            loss = criterion(pred, y)

        metrics = compute_all_metrics(
            pred=pred,
            target=y,
            data_range=data_range,
            clamp_for_metrics=clamp_for_metrics,
        )

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_mae += metrics["mae"] * bs
        total_nmse += metrics["nmse"] * bs
        total_ssim += metrics["ssim"] * bs
        total_psnr += metrics["psnr"] * bs
        n += bs

    if n == 0:
        raise ValueError("DataLoader produced no batches")

    return {
        "loss": total_loss / max(n, 1),
        "mae": total_mae / max(n, 1),
        "nmse": total_nmse / max(n, 1),
        "ssim": total_ssim / max(n, 1),
        "psnr": total_psnr / max(n, 1),
    }


# =========================================================
# 6. Run
# =========================================================
def main(cfg=None):
    import sys
    from runner import run_training
    return run_training('unet', sys.modules[__name__], cfg or Config())


if __name__ == '__main__':
    main(parse_config(Config))
