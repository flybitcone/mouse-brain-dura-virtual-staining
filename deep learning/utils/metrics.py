"""Image-wise metrics, averaged over images (independent of batch partitioning)."""
import torch
from pytorch_msssim import ssim


def prepare_for_metrics(pred, target, data_range=1.0, clamp_for_metrics=True):
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError(f"Expected equal NCHW tensors: {pred.shape}, {target.shape}")
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    pred, target = pred.float(), target.float()
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError("Non-finite prediction or target")
    if (target < 0).any() or (target > data_range).any():
        raise ValueError("Target outside configured data_range; check normalization")
    if clamp_for_metrics:
        pred = pred.clamp(0, data_range)
    return pred, target


def compute_mae(pred, target):
    return (pred.float() - target.float()).abs().mean().item()


def compute_nmse(pred, target):
    # Epsilon convention: blank targets with false positives must not score zero.
    mse = (pred.float() - target.float()).square().flatten(1).mean(1)
    energy = target.float().square().flatten(1).mean(1)
    return (mse / energy.clamp_min(1e-12)).mean().item()


def compute_psnr(pred, target, data_range=1.0):
    mse = (pred.float() - target.float()).square().flatten(1).mean(1)
    # Finite 120 dB ceiling for exact matches at data_range=1.
    return (10 * torch.log10(data_range ** 2 / mse.clamp_min(1e-12))).mean().item()


def compute_ssim(pred, target, data_range=1.0):
    with torch.autocast(device_type=pred.device.type, enabled=False):
        return ssim(pred.float(), target.float(), data_range=data_range,
                    size_average=True).item()


@torch.no_grad()
def compute_all_metrics(pred, target, data_range=1.0, clamp_for_metrics=True):
    return {k: v.mean().item() for k, v in compute_per_image_metrics(
        pred, target, data_range, clamp_for_metrics).items()}


@torch.no_grad()
def compute_per_image_metrics(pred, target, data_range=1.0, clamp_for_metrics=True):
    pred, target = prepare_for_metrics(pred, target, data_range, clamp_for_metrics)
    error = (pred-target).flatten(1)
    mse = error.square().mean(1)
    energy = target.square().flatten(1).mean(1)
    with torch.autocast(device_type=pred.device.type, enabled=False):
        structural = ssim(pred, target, data_range=data_range, size_average=False)
    return dict(mae=error.abs().mean(1), nmse=mse/energy.clamp_min(1e-12),
                psnr=10*torch.log10(data_range**2/mse.clamp_min(1e-12)), ssim=structural)
