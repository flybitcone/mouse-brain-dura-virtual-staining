"""Validation-only equal-budget learning-rate/loss-weight search. Never scores test data."""
import argparse
import importlib
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path
import torch
from eval import MODULES
from utils.training import create_run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODULES, required=True)
    parser.add_argument('--task', default='BFtoIBA1')
    parser.add_argument('--data-root', default=str(Path(__file__).resolve().parent/'dataset_patches'))
    parser.add_argument('--output-root', default=str(Path(__file__).resolve().parent/'tuning_runs'))
    parser.add_argument('--lrs', nargs='+', type=float, default=[1e-4,2e-4,4e-4])
    parser.add_argument('--ssim-weights', nargs='+', type=float, default=[0.1,0.2])
    parser.add_argument('--trial-epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if args.trial_epochs < 1:
        parser.error('trial-epochs must be positive')
    module = importlib.import_module(MODULES[args.model])
    root = create_run_dir(args.output_root, args.task, args.model)
    results = []
    weights = [0.2] if args.model in ('pix2pix','cyclegan') else args.ssim_weights
    for lr, ssim_weight in product(args.lrs, weights):
        cfg = module.Config()
        cfg.task, cfg.data_root, cfg.output_root = args.task, args.data_root, str(root/'trials')
        cfg.epochs = cfg.schedule_epochs = args.trial_epochs
        cfg.warmup_epochs, cfg.patience = min(3, args.trial_epochs-1), 0
        cfg.batch_size, cfg.seed, cfg.device = args.batch_size, args.seed, args.device
        cfg.test_after_train = False
        cfg.lambda_ssim = ssim_weight
        if hasattr(cfg, 'lr_G'):
            cfg.lr_G = cfg.lr_D = lr
        else:
            cfg.lr = lr
        run = module.main(cfg)
        checkpoint = torch.load(run/'checkpoints/best_model.pt', weights_only=True, map_location='cpu')
        results.append(dict(lr=lr, lambda_ssim=ssim_weight, val_mae=checkpoint['best_val_mae'], run=str(run), config=asdict(cfg)))
        (root/'trials.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    best = min(results, key=lambda row:row['val_mae'])
    recommended = dict(best['config'])
    recommended.update(epochs=300, schedule_epochs=300, warmup_epochs=5, patience=50,
                       test_after_train=True, output_root=str(Path(__file__).resolve().parent/'runs'))
    (root/'selected_config.json').write_text(json.dumps(recommended, indent=2), encoding='utf-8')
    print(json.dumps(dict(selected=best['run'], selection_metric='validation MAE',
                         note='Short-budget selection, not a guarantee of the full-training optimum.',
                         config=str(root/'selected_config.json')), indent=2))


if __name__ == '__main__':
    main()
