import os
import torch


def save_checkpoint(path: str, payload: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, device: str = "cpu"):
    return torch.load(path, map_location=device, weights_only=True)
