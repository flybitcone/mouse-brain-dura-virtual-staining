from dataclasses import dataclass
import sys
from Unet_train import Config as BaseConfig, CombinedLoss, train_one_epoch, validate_one_epoch
from models.restormer_staining import Restormer
from utils.training import parse_config


@dataclass
class Config(BaseConfig):
    batch_size: int = 2
    dim: int = 48
    num_blocks: tuple = (4, 6, 6, 8)
    num_refinement_blocks: int = 4
    heads: tuple = (1, 2, 4, 8)


def build_model(cfg):
    if cfg.dim % 2 or any((cfg.dim * 2**i) % h for i, h in enumerate(cfg.heads)):
        raise ValueError('Restormer dimensions must be even and divisible by attention heads')
    return Restormer(inp_channels=cfg.in_channels, out_channels=cfg.out_channels, dim=cfg.dim,
                     num_blocks=cfg.num_blocks, num_refinement_blocks=cfg.num_refinement_blocks, heads=cfg.heads)


def main(cfg=None):
    from runner import run_training
    return run_training('restormer', sys.modules[__name__], cfg or Config())


if __name__ == '__main__':
    main(parse_config(Config))
