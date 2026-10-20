"""Evaluate checkpoints produced by any of the five training scripts."""
import argparse
import csv
from collections import defaultdict
import importlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset import VirtualStainingPatchDataset, subject_from_key
from utils.metrics import compute_per_image_metrics
from utils.training import choose_device, check_global_splits, dataset_signature

MODULES = {
    'unet': 'Unet_train', 'resunet': 'ResUnet_train',
    'attunet': 'AttUnet_train', 'pix2pix': 'Pix2pixcGAN_train',
    'cyclegan': 'CycleGAN_train',
    'unetpp': 'UNetPlusPlus_train', 'restormer': 'Restormer_train', 'swinir': 'SwinIR_train',
}


def tensor_to_uint8_img(t):
    t = t.detach().cpu().float().clamp(0, 1).mul(255).round().byte()
    if t.ndim == 3:
        if t.shape[0] == 1:
            t = t[0]
        elif t.shape[0] == 3:
            t = t.permute(1, 2, 0)
        else:
            raise ValueError('Image export supports one or three channels')
    if t.ndim not in (2, 3):
        raise ValueError('Expected CHW or HW image')
    return Image.fromarray(t.numpy())


def load_model(model_name, checkpoint, device):
    if checkpoint.get('model_name', model_name) != model_name:
        raise ValueError('Checkpoint model differs from --model')
    module = importlib.import_module(MODULES[model_name])
    valid = {f.name for f in fields(module.Config)}
    cfg = module.Config(**{k: v for k, v in checkpoint.get('config', {}).items() if k in valid})
    if model_name == 'cyclegan':
        model = module.ResNetGenerator(cfg.in_channels_A, cfg.in_channels_B, cfg.n_residual_blocks, cfg.base_channels)
        key = 'G_AB_state_dict'
    elif model_name == 'pix2pix':
        model = module.GeneratorUNet(cfg.in_channels, cfg.out_channels, cfg.base_channels)
        key = 'generator_state_dict'
    else:
        model = module.build_model(cfg)
        key = 'model_state_dict'
    model.load_state_dict(checkpoint[key], strict=True)
    return model.to(device), cfg


@torch.no_grad()
def evaluate(model, loader, device, save_dir=None, save_max=50, clamp_for_metrics=True, report_dir=None):
    model.eval()
    names = ('mae', 'nmse', 'psnr', 'ssim')
    records = []
    saved = 0
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    for batch in loader:
        x, y = batch['input'].to(device), batch['target'].to(device)
        # Models have their own output activation; never apply an extra sigmoid.
        pred = model(x)
        metrics = {k:v.cpu().tolist() for k,v in compute_per_image_metrics(pred, y, clamp_for_metrics=clamp_for_metrics).items()}
        for i, patch_id in enumerate(batch['patch_id']):
            subject = batch['subject_id'][i] if 'subject_id' in batch else subject_from_key(batch['fov_key'][i])
            records.append(dict(subject_id=subject, fov_key=batch['fov_key'][i], patch_id=patch_id,
                                **{k:metrics[k][i] for k in names}))
        if save_dir is not None:
            for i in range(min(x.shape[0], max(0, save_max - saved))):
                stem = batch['patch_id'][i]
                for kind, tensor in (('input', x[i]), ('target', y[i]), ('pred', pred[i])):
                    tensor_to_uint8_img(tensor).save(save_dir / f'{stem}_{kind}.png')
                export = pred[i].float().clamp(0, 1) if clamp_for_metrics else pred[i].float()
                np.save(save_dir / f'{stem}_pred.npy', export.cpu().numpy())
                saved += 1
    if not records:
        raise ValueError('Empty evaluation loader')
    def means(rows):
        return {k:float(np.mean([r[k] for r in rows], dtype=np.float64)) for k in names}
    grouped = defaultdict(list)
    for row in records:
        grouped[(row['subject_id'], row['fov_key'])].append(row)
    fovs = [dict(subject_id=s, fov_key=f, patches=len(rows), **means(rows)) for (s,f),rows in sorted(grouped.items())]
    grouped = defaultdict(list)
    for row in fovs:
        grouped[row['subject_id']].append(row)
    mice = [dict(subject_id=s, fovs=len(rows), patches=sum(r['patches'] for r in rows), **means(rows)) for s,rows in sorted(grouped.items())]
    result = dict(samples=len(records), fovs=len(fovs), mice=len(mice), metrics=means(records),
                  macro_fov=means(fovs), macro_mouse=means(mice),
                  aggregation='FOV: mean patch metrics; mouse: mean FOV metrics. Not stitched-image metrics.',
                  clamp_for_metrics=clamp_for_metrics)
    if report_dir is not None:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        for filename, rows in [('per_patch.csv', records), ('per_fov.csv', fovs), ('per_mouse.csv', mice)]:
            with (report_dir / filename).open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (report_dir / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=MODULES)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--data-root', type=Path, default=Path(__file__).parent / 'dataset_patches')
    parser.add_argument('--task', default=None)
    parser.add_argument('--subset', choices=('val', 'test'), default='test')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--save-max', type=int, default=50)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--max-batches', type=int, default=0, help='Debug limit; 0 means full subset')
    parser.add_argument('--allow-legacy-checkpoint', action='store_true')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0 or args.save_max < 0:
        parser.error('Invalid batch-size, num-workers or save-max')
    if args.max_batches < 0 or args.cpu_threads < 1:
        parser.error('Invalid max-batches or cpu-threads')
    torch.set_num_threads(args.cpu_threads)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, cfg = load_model(args.model, checkpoint, device)
    task = args.task or cfg.task
    if task != cfg.task:
        raise ValueError('Checkpoint task differs from requested task')
    check_global_splits(args.data_root)
    signature = dataset_signature(args.data_root, task)
    verified = 'data_signature' in checkpoint
    if verified and checkpoint['data_signature'] != signature:
        raise ValueError('Checkpoint data signature differs from the current dataset')
    if not verified and not args.allow_legacy_checkpoint:
        raise ValueError('Legacy checkpoint lacks data provenance; explicitly pass --allow-legacy-checkpoint')
    dataset = VirtualStainingPatchDataset(args.data_root, task, args.subset, check_files=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, pin_memory=device.type == 'cuda')
    from runner import LimitedLoader
    loader = LimitedLoader(loader, args.max_batches)
    from utils.training import create_run_dir
    if args.output is None:
        output = create_run_dir(args.checkpoint.parent.parent, 'evaluation', args.subset)
    else:
        output = args.output
        output.mkdir(parents=True, exist_ok=False)
    result = evaluate(model, loader, device, output / 'predictions' if args.save_max else None,
                      args.save_max, cfg.clamp_for_metrics, report_dir=output)
    result.update(checkpoint=str(args.checkpoint.resolve()), model=args.model,
                  task=dataset.task, subset=args.subset, data_signature=signature, provenance_verified=verified,
                  limited_run=bool(args.max_batches or checkpoint.get('limited_run', False)))
    (output / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return output


if __name__ == '__main__':
    main()
