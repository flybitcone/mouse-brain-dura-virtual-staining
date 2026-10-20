"""Shared configuration and preflight checks for training entry points."""
import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass, fields
from itertools import combinations
from pathlib import Path


@dataclass
class RunConfig:
    device: str = 'auto'
    resume: str = ''
    test_after_train: bool = True
    max_train_batches: int = 0
    max_val_batches: int = 0
    max_test_batches: int = 0
    cpu_threads: int = 4
    augment: bool = True
    grad_clip: float = 1.0
    scheduler: str = 'cosine'
    warmup_epochs: int = 5
    schedule_epochs: int = 300
    min_lr: float = 1e-6
    patience: int = 50
    lambda_ssim: float = 0.2


def choose_device(name='auto'):
    import torch
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Supported devices: auto, cpu, cuda, cuda:N')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable in this Python environment')
    return device


def dataset_signature(root, task):
    result = {}
    for subset in ('train', 'val', 'test'):
        path = Path(root) / task / subset / 'metadata.csv'
        if not path.is_file():
            raise FileNotFoundError(f'Benchmark metadata required: {path}')
        result[subset] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def check_global_splits(root):
    """Metadata only; never scan all image pixels."""
    from dataset import subject_from_key
    ownership = {}
    for metadata in sorted(Path(root).glob('*/[tval]*/metadata.csv')):
        subset = metadata.parent.name
        if subset not in ('train', 'val', 'test'):
            continue
        with metadata.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                subject = row.get('subject_id') or subject_from_key(row['fov_key'])
                if subject_from_key(row['fov_key']) != subject:
                    raise ValueError(f'Subject/FOV mismatch: {metadata}')
                if row.get('subset', subset) != subset or row.get('task', metadata.parent.parent.name) != metadata.parent.parent.name:
                    raise ValueError(f'Task/subset mismatch: {metadata}')
                if subject in ownership and ownership[subject] != subset:
                    raise ValueError(f'Global data leakage: {subject}: {ownership[subject]}/{subset}')
                ownership[subject] = subset
    if not ownership:
        raise ValueError('No split metadata found')
    return ownership


def parse_config(config_type):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default='')
    pre.add_argument('--resume', default='')
    initial, _ = pre.parse_known_args()
    if initial.config and initial.resume:
        pre.error('Use either --config or --resume, not both')
    parser = argparse.ArgumentParser()
    defaults = config_type()
    overrides = {}
    if initial.config:
        overrides = json.loads(Path(initial.config).read_text(encoding='utf-8'))
        overrides = overrides.get('config', overrides)
    elif initial.resume:
        import torch
        overrides = torch.load(initial.resume, map_location='cpu', weights_only=True)['config']
        overrides['resume'] = initial.resume
    known = {f.name for f in fields(defaults)}
    if set(overrides) - known:
        parser.error(f'Unknown configuration keys: {sorted(set(overrides)-known)}')
    for key, value in overrides.items():
        setattr(defaults, key, value)
    parser.add_argument('--config', default=initial.config)
    for field in fields(defaults):
        value = getattr(defaults, field.name)
        if isinstance(value, (tuple, list)):
            parser.add_argument('--' + field.name.replace('_', '-'), nargs=len(value),
                                type=int, default=value)
        elif isinstance(value, bool):
            parser.add_argument('--' + field.name.replace('_', '-'),
                                action=argparse.BooleanOptionalAction, default=value)
        else:
            parser.add_argument('--' + field.name.replace('_', '-'), type=type(value), default=value)
    arguments = vars(parser.parse_args())
    arguments.pop('config')
    return config_type(**arguments)


def validate_config(cfg):
    for name in ('epochs', 'batch_size', 'save_sample_every', 'num_sample_images',
                 'lr', 'lr_G', 'lr_D', 'base_channels', 'in_channels', 'out_channels',
                 'in_channels_A', 'in_channels_B', 'image_size'):
        value = getattr(cfg, name, None)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f'{name} must be finite and positive')
    if cfg.num_workers < 0:
        raise ValueError('num_workers must be nonnegative')
    if cfg.data_range != 1.0:
        raise ValueError('This dataset and GAN output activation require data_range=1.0')
    for name in ('weight_decay', 'lambda_l1', 'lambda_cycle', 'lambda_identity', 'lambda_ssim', 'pool_size'):
        value = getattr(cfg, name, 0)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    for name in ('beta1', 'beta2'):
        if not 0 <= getattr(cfg, name, 0) < 1:
            raise ValueError(f'{name} must be in [0, 1)')
    if getattr(cfg, 'lambda_identity', 0) and cfg.in_channels_A != cfg.in_channels_B:
        raise ValueError('Identity loss requires equal A/B channel counts; use --lambda-identity 0')
    for name in ('max_train_batches', 'max_val_batches', 'max_test_batches'):
        if getattr(cfg, name, 0) < 0:
            raise ValueError(f'{name} must be nonnegative')
    if getattr(cfg, 'cpu_threads', 1) < 1:
        raise ValueError('cpu_threads must be positive')
    if hasattr(cfg, 'features') and (len(cfg.features) != 6 or min(cfg.features) < 1):
        raise ValueError('BasicUNet features must contain six positive integers')
    if hasattr(cfg, 'n_residual_blocks') and cfg.n_residual_blocks < 1:
        raise ValueError('n_residual_blocks must be positive')
    if cfg.scheduler not in ('cosine', 'constant') or not 0 <= cfg.warmup_epochs < cfg.schedule_epochs:
        raise ValueError('Invalid scheduler or warmup/schedule epoch counts')
    if cfg.patience < 0 or not math.isfinite(cfg.grad_clip) or cfg.grad_clip < 0 or not 0 <= cfg.min_lr <= min(getattr(cfg, 'lr', float('inf')), getattr(cfg, 'lr_G', float('inf')), getattr(cfg, 'lr_D', float('inf'))):
        raise ValueError('Invalid patience, grad-clip or min-lr')
    for name in ('dim', 'embed_dim', 'window_size', 'num_refinement_blocks'):
        if hasattr(cfg, name) and getattr(cfg, name) < 1:
            raise ValueError(f'{name} must be positive')
    for name in ('num_blocks', 'heads', 'depths', 'num_heads'):
        if hasattr(cfg, name) and (not getattr(cfg, name) or min(getattr(cfg, name)) < 1):
            raise ValueError(f'{name} must contain positive values')


def clip_gradients(scaler, optimizer, maximum):
    if maximum:
        import torch
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params']], maximum,
                                      error_if_nonfinite=False)


def create_run_dir(output_root, task, model_name):
    # Never mix a fresh run with historical checkpoints or truncate old logs.
    base = Path(output_root) / task / model_name
    candidate = base
    index = 1
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = base.with_name(f'{model_name}_{index:03d}')
            index += 1


def check_datasets(cfg, *datasets):
    from dataset import subject_from_key
    datasets = [getattr(ds, 'base_ds', ds) for ds in datasets]
    for ds in datasets:
        x, y = ds[0] if not ds.return_metadata else (ds[0]['input'], ds[0]['target'])
        in_c = getattr(cfg, 'in_channels', getattr(cfg, 'in_channels_A', 3))
        out_c = getattr(cfg, 'out_channels', getattr(cfg, 'in_channels_B', 1))
        if x.shape[0] != in_c or y.shape[0] != out_c:
            raise ValueError(f'{ds.subset}: configured channels do not match data')
        if hasattr(cfg, 'image_size') and x.shape[-2:] != (cfg.image_size, cfg.image_size):
            raise ValueError(f'{ds.subset}: image_size does not match data')
    for left, right in combinations(datasets, 2):
        for kind in ('patch_id', 'fov_key', 'subject'):
            def keys(ds):
                result = set()
                for sample in ds.samples:
                    fov = sample['fov_key'] or re.sub(r'_r\d+_c\d+$', '', sample['patch_id'])
                    key = (sample.get('subject_id') or subject_from_key(fov)) if kind == 'subject' else (fov if kind == 'fov_key' else sample[kind])
                    result.add(key)
                return result
            overlap = keys(left) & keys(right)
            if overlap:
                raise ValueError(f'Data leakage: {left.subset}/{right.subset} share {kind}: {sorted(overlap)[:5]}')
