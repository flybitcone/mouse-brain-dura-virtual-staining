# Third-party components

The MIT license covers the authors' original code. It does not replace third-party licenses.
Vendored Restormer files retain `deep learning/models/restormer_LICENSE.txt`;
vendored SwinIR retains `deep learning/models/swinir_LICENSE.txt`.
Pinned upstream commits and source checksums are in `deep learning/models/upstream_sources.json`.
The Restormer staining adapter is identified in its source header.
Installed dependencies including PyTorch and MONAI retain their respective licenses.

Model references:
- U-Net: https://arxiv.org/abs/1505.04597
- Residual U-Net: https://arxiv.org/abs/1711.10684
- Attention U-Net: https://arxiv.org/abs/1804.03999
- pix2pix: https://arxiv.org/abs/1611.07004
- CycleGAN: https://arxiv.org/abs/1703.10593
- Restormer: https://github.com/swz30/Restormer
- SwinIR: https://github.com/JingyunLiang/SwinIR

These scripts contain task-specific implementations/adaptations; a model name does not
imply bitwise reproduction of its original paper. Additional model scripts are included
for completeness and are not all part of the five-model manuscript benchmark.
