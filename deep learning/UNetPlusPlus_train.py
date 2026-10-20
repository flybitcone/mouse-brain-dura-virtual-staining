from dataclasses import dataclass
import sys
import torch.nn as nn
from monai.networks.nets import BasicUNetPlusPlus
from Unet_train import Config as BaseConfig, CombinedLoss, train_one_epoch, validate_one_epoch
from utils.training import parse_config


@dataclass
class Config(BaseConfig):
    batch_size: int = 4


class NestedUNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.network = BasicUNetPlusPlus(spatial_dims=2, in_channels=cfg.in_channels,
            out_channels=cfg.out_channels, features=cfg.features, deep_supervision=False,
            norm=('INSTANCE', {'affine': True}), upsample='nontrainable')

    def forward(self, x):
        return self.network(x)[0]


def build_model(cfg):
    return NestedUNet(cfg)


def main(cfg=None):
    from runner import run_training
    return run_training('unetpp', sys.modules[__name__], cfg or Config())


if __name__ == '__main__':
    main(parse_config(Config))
