from dataclasses import dataclass
import sys
import torch.nn as nn
from Unet_train import Config as BaseConfig, CombinedLoss, train_one_epoch, validate_one_epoch
from models.swinir_upstream import SwinIR
from utils.training import parse_config


@dataclass
class Config(BaseConfig):
    batch_size: int = 2
    image_size: int = 256
    embed_dim: int = 60
    depths: tuple = (6, 6, 6, 6)
    num_heads: tuple = (6, 6, 6, 6)
    window_size: int = 8
    use_checkpoint: bool = True


class SwinIRForStaining(SwinIR):
    """Official same-resolution backbone with a 3->1 regression head.

    RGB restoration mean and raw-image residual are omitted for cross-modality prediction.
    Feature residuals and shifted-window attention remain unchanged.
    """
    def __init__(self, cfg):
        super().__init__(img_size=cfg.image_size, in_chans=cfg.in_channels, embed_dim=cfg.embed_dim,
            depths=cfg.depths, num_heads=cfg.num_heads, window_size=cfg.window_size,
            upscale=1, upsampler='', img_range=1., use_checkpoint=cfg.use_checkpoint)
        self.conv_last = nn.Conv2d(cfg.embed_dim, cfg.out_channels, 3, 1, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = self.check_image_size(x)
        shallow = self.conv_first(x)
        features = self.conv_after_body(self.forward_features(shallow)) + shallow
        return self.conv_last(features)[..., :h, :w]


def build_model(cfg):
    if len(cfg.depths) != len(cfg.num_heads) or any(cfg.embed_dim % h for h in cfg.num_heads):
        raise ValueError('SwinIR depths/heads must agree and divide embed_dim')
    return SwinIRForStaining(cfg)


def main(cfg=None):
    from runner import run_training
    return run_training('swinir', sys.modules[__name__], cfg or Config())


if __name__ == '__main__':
    main(parse_config(Config))
