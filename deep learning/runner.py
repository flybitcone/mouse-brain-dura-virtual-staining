"""Common training lifecycle for the five baseline architectures."""
import csv
import json
import math
import random
import shutil
from dataclasses import asdict
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import VirtualStainingPatchDataset
from utils.seed import seed_everything
from utils.training import (choose_device, check_datasets, check_global_splits,
                            create_run_dir, dataset_signature, validate_config)


class LimitedLoader:
    def __init__(self, loader, limit=0):
        self.loader, self.limit = loader, limit

    def __iter__(self):
        return islice(self.loader, self.limit) if self.limit else iter(self.loader)

    def __len__(self):
        return min(len(self.loader), self.limit) if self.limit else len(self.loader)


class PairedAugment:
    def __call__(self, x, y):
        k = random.randrange(4)
        x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
        if random.random() < 0.5:
            x, y = x.flip(-1), y.flip(-1)
        return x, y


def make_scheduler(optimizer, cfg):
    base_lr = optimizer.param_groups[0]['lr']
    def factor(epoch_index):
        if cfg.scheduler == 'constant':
            return 1.0
        if epoch_index < cfg.warmup_epochs:
            return (epoch_index+1) / max(1, cfg.warmup_epochs)
        progress = min(1.0, (epoch_index-cfg.warmup_epochs) / (cfg.schedule_epochs-cfg.warmup_epochs))
        minimum = cfg.min_lr/base_lr
        return minimum + (1-minimum)*0.5*(1+math.cos(math.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def atomic_save(payload, path):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def rng_state(generators):
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=(state[0], state[1].tolist(), *state[2:]),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                loaders={k: g.get_state() for k, g in generators.items()})


def restore_rng(state, generators):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    for key, generator in generators.items():
        generator.set_state(state['loaders'][key])


def setup(module, name, cfg, device):
    models, optimizers, scalers, pools = {}, {}, {}, {}
    if name == 'cyclegan':
        models = {
            'G_AB_state_dict': module.ResNetGenerator(cfg.in_channels_A, cfg.in_channels_B, cfg.n_residual_blocks, cfg.base_channels),
            'G_BA_state_dict': module.ResNetGenerator(cfg.in_channels_B, cfg.in_channels_A, cfg.n_residual_blocks, cfg.base_channels),
            'D_A_state_dict': module.Discriminator(cfg.in_channels_A, cfg.base_channels),
            'D_B_state_dict': module.Discriminator(cfg.in_channels_B, cfg.base_channels)}
        criterion = module.CycleGANLoss(cfg.lambda_cycle, cfg.lambda_identity)
        pools = {key: module.ImagePool(cfg.pool_size) for key in ('fake_A_pool', 'fake_B_pool')}
    elif name == 'pix2pix':
        models = {'generator_state_dict': module.GeneratorUNet(cfg.in_channels, cfg.out_channels, cfg.base_channels),
                  'discriminator_state_dict': module.Discriminator(cfg.in_channels, cfg.out_channels, cfg.base_channels)}
        criterion = module.Pix2PixLoss(cfg.lambda_l1)
    else:
        models = {'model_state_dict': module.build_model(cfg)}
        criterion = module.CombinedLoss(lambda_l1=1.0, lambda_ssim=cfg.lambda_ssim)
    for model in models.values():
        model.to(device)
        if name in ('pix2pix', 'cyclegan'):
            module.init_weights(model)
    if name == 'cyclegan':
        params = list(models['G_AB_state_dict'].parameters()) + list(models['G_BA_state_dict'].parameters())
        optimizers['optimizer_G_state_dict'] = torch.optim.Adam(params, lr=cfg.lr_G, betas=(cfg.beta1, cfg.beta2))
        for domain in ('A', 'B'):
            optimizers[f'optimizer_D_{domain}_state_dict'] = torch.optim.Adam(
                models[f'D_{domain}_state_dict'].parameters(), lr=cfg.lr_D, betas=(cfg.beta1, cfg.beta2))
    elif name == 'pix2pix':
        for label, key, lr in [('G', 'generator_state_dict', cfg.lr_G), ('D', 'discriminator_state_dict', cfg.lr_D)]:
            optimizers[f'optimizer_{label}_state_dict'] = torch.optim.Adam(models[key].parameters(), lr=lr, betas=(cfg.beta1, cfg.beta2))
    else:
        optimizers['optimizer_state_dict'] = torch.optim.AdamW(models['model_state_dict'].parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for key in optimizers:
        scalers[key.replace('optimizer', 'scaler')] = torch.amp.GradScaler('cuda', enabled=cfg.amp and device.type == 'cuda')
    return models, optimizers, scalers, pools, criterion


def epoch(module, name, cfg, parts, loader, device, training):
    models, optimizers, scalers, pools, criterion = parts
    kw = dict(loader=loader, criterion=criterion, device=device, data_range=cfg.data_range,
              clamp_for_metrics=cfg.clamp_for_metrics, use_amp=cfg.amp and device.type == 'cuda')
    if training:
        kw['grad_clip'] = cfg.grad_clip
    if name == 'cyclegan':
        for key in ('G_AB', 'G_BA', 'D_A', 'D_B'):
            kw[key] = models[key + '_state_dict']
        if training:
            kw.update(pools)
            for key, value in optimizers.items():
                kw[key.removesuffix('_state_dict')] = value
            for key, value in scalers.items():
                kw[key.removesuffix('_state_dict')] = value
    elif name == 'pix2pix':
        kw.update(generator=models['generator_state_dict'], discriminator=models['discriminator_state_dict'])
        if training:
            for key, value in {**optimizers, **scalers}.items():
                kw[key.removesuffix('_state_dict')] = value
    else:
        kw['model'] = models['model_state_dict']
        if training:
            kw.update(optimizer=optimizers['optimizer_state_dict'], scaler=scalers['scaler_state_dict'])
    result = (module.train_one_epoch if training else module.validate_one_epoch)(**kw)
    if not all(math.isfinite(v) for v in result.values()):
        raise FloatingPointError('Non-finite epoch metrics; checkpoint not saved')
    return result


@torch.no_grad()
def preview(model, loader, destination, maximum):
    from PIL import Image
    from eval import tensor_to_uint8_img
    previous = model.training
    model.eval()
    batch = next(iter(loader))
    x = batch.get('input', batch.get('A'))
    y = batch.get('target', batch.get('B_pair'))
    count = min(maximum, x.shape[0])
    pred = model(x[:count].to(next(model.parameters()).device)).cpu()
    h, w = x.shape[-2:]
    canvas = Image.new('RGB', (3*w, count*h))
    for i in range(count):
        for j, tensor in enumerate((x[i], y[i], pred[i])):
            canvas.paste(tensor_to_uint8_img(tensor).convert('RGB'), (j*w, i*h))
    canvas.save(destination)
    model.train(previous)


def run_training(name, module, cfg):
    validate_config(cfg)
    device = choose_device(cfg.device)
    torch.set_num_threads(cfg.cpu_threads)
    seed_everything(cfg.seed)
    ownership = check_global_splits(cfg.data_root)
    signature = dataset_signature(cfg.data_root, cfg.task)
    base = {s: VirtualStainingPatchDataset(cfg.data_root, cfg.task, s, check_files=False,
                joint_transform=PairedAugment() if cfg.augment and s == 'train' else None)
            for s in ('train', 'val', 'test')}
    check_datasets(cfg, *base.values())
    # Pixel reads are restricted to batches consumed by training/evaluation.
    datasets = dict(base)
    if name == 'cyclegan':
        for s in ('train', 'val'):
            wrapper = module.UnpairedVirtualStainingDataset.__new__(module.UnpairedVirtualStainingDataset)
            wrapper.base_ds, wrapper.unpaired, wrapper.subset = base[s], s == 'train', s
            datasets[s] = wrapper
    generators = {s: torch.Generator().manual_seed(cfg.seed+i) for i, s in enumerate(base)}
    loaders = {s: LimitedLoader(DataLoader(ds, batch_size=cfg.batch_size, shuffle=s == 'train',
                    num_workers=cfg.num_workers, pin_memory=device.type == 'cuda', drop_last=False,
                    generator=generators[s], worker_init_fn=seed_worker), getattr(cfg, f'max_{s}_batches'))
               for s, ds in datasets.items()}
    parts = setup(module, name, cfg, device)
    models, optimizers, scalers, pools, criterion = parts
    schedulers = {k:make_scheduler(opt, cfg) for k,opt in optimizers.items()}
    predictor_key = {'pix2pix':'generator_state_dict', 'cyclegan':'G_AB_state_dict'}.get(name, 'model_state_dict')
    predictor = models[predictor_key]
    checkpoint = None
    start_epoch, best_mae = 1, float('inf')
    bad_epochs = 0
    if cfg.resume:
        checkpoint = torch.load(cfg.resume, map_location='cpu', weights_only=True)
        if checkpoint.get('format_version') != 2 or checkpoint.get('model_name') != name:
            raise ValueError('Resume requires a matching version-2 training checkpoint')
        if checkpoint['data_signature'] != signature:
            raise ValueError('Dataset metadata changed; refuse to resume across different splits/preprocessing')
        allowed = {'resume', 'epochs', 'output_root', 'device', 'cpu_threads', 'test_after_train', 'save_sample_every', 'num_sample_images', 'data_root'}
        old = checkpoint['config']
        for key, value in asdict(cfg).items():
            if key not in allowed and json.dumps(value) != json.dumps(old[key]):
                raise ValueError(f'Resume configuration changed: {key}')
        start_epoch, best_mae = checkpoint['epoch']+1, checkpoint['best_val_mae']
        bad_epochs = checkpoint.get('bad_epochs', 0)
        if start_epoch > cfg.epochs:
            raise ValueError('--epochs is the total epoch count and must exceed the saved epoch')
        for key, value in {**models, **optimizers, **scalers}.items():
            value.load_state_dict(checkpoint[key])
        for key, scheduler in schedulers.items():
            scheduler.load_state_dict(checkpoint['schedulers'][key])
        for key, pool in pools.items():
            pool.images.extend(t.to(device) for t in checkpoint['pools'][key])
        restore_rng(checkpoint['rng'], generators)
    run_dir = create_run_dir(cfg.output_root, cfg.task, name)
    (run_dir / 'checkpoints').mkdir()
    (run_dir / 'samples').mkdir()
    config = asdict(cfg)
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (run_dir / 'data_signature.json').write_text(json.dumps(signature, indent=2), encoding='utf-8')
    (run_dir / 'split_subjects.json').write_text(json.dumps(ownership, indent=2), encoding='utf-8')
    (run_dir / 'environment.json').write_text(json.dumps(dict(torch=str(torch.__version__), numpy=str(np.__version__), device=str(device)), indent=2), encoding='utf-8')
    best_path, last_path = (run_dir / 'checkpoints' / f'{kind}_model.pt' for kind in ('best', 'last'))
    if checkpoint is not None:
        previous_best = Path(cfg.resume).parent / 'best_model.pt'
        if not previous_best.is_file():
            raise FileNotFoundError('Best checkpoint next to resume checkpoint is required')
        best_checkpoint = torch.load(previous_best, map_location='cpu', weights_only=True)
        if best_checkpoint.get('model_name') != name or best_checkpoint.get('data_signature') != signature or best_checkpoint['epoch'] > checkpoint['epoch']:
            raise ValueError('Incompatible best checkpoint next to resume checkpoint')
        shutil.copy2(previous_best, best_path)
        del best_checkpoint, checkpoint
    limited = any(getattr(cfg, f'max_{s}_batches') for s in base)
    print(f'[INFO] {name}/{cfg.task} device={device} run={run_dir} limited_run={limited}', flush=True)
    for number in range(start_epoch, cfg.epochs+1):
        train = epoch(module, name, cfg, parts, loaders['train'], device, True)
        val = epoch(module, name, cfg, parts, loaders['val'], device, False)
        row = dict(epoch=number, **{'lr_'+k.removesuffix('_state_dict'):v.param_groups[0]['lr'] for k,v in optimizers.items()},
                   **{'train_'+k:v for k,v in train.items()}, **{'val_'+k:v for k,v in val.items()})
        log = run_dir / 'train_log.csv'
        exists = log.exists()
        with log.open('a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        improved = val['mae'] < best_mae
        bad_epochs = 0 if improved else bad_epochs+1
        best_mae = min(best_mae, val['mae'])
        if number == 1 or number % cfg.save_sample_every == 0:
            preview(predictor, loaders['val'], run_dir / 'samples' / f'epoch_{number:04}.png', cfg.num_sample_images)
        for scheduler in schedulers.values():
            scheduler.step()
        payload = dict(format_version=2, model_name=name, config=config, epoch=number,
                       best_val_mae=best_mae, selection_metric='val_mae', val_metrics=val,
                       data_signature=signature, limited_run=limited, rng=rng_state(generators),
                       bad_epochs=bad_epochs, schedulers={k:v.state_dict() for k,v in schedulers.items()},
                       pools={k:[t.detach().cpu() for t in pool.images] for k,pool in pools.items()})
        payload.update({k:v.state_dict() for k,v in {**models, **optimizers, **scalers}.items()})
        atomic_save(payload, last_path)
        if improved:
            shutil.copy2(last_path, best_path)
        print(f'[Epoch {number}/{cfg.epochs}] train_mae={train["mae"]:.6f} val_mae={val["mae"]:.6f} best={best_mae:.6f}', flush=True)
        if cfg.patience and bad_epochs >= cfg.patience:
            print(f'[INFO] Early stopping after {cfg.patience} epochs without validation improvement', flush=True)
            break
    if cfg.test_after_train:
        from eval import evaluate
        best = torch.load(best_path, map_location='cpu', weights_only=True)
        predictor.load_state_dict(best[predictor_key])
        report = evaluate(predictor, loaders['test'], device, report_dir=run_dir / 'test', clamp_for_metrics=cfg.clamp_for_metrics)
        report.update(model=name, task=cfg.task, subset='test', checkpoint=str(best_path),
                      best_epoch=best['epoch'], limited_run=limited, data_signature=signature)
        (run_dir / 'test_metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(f'[TEST] {report["metrics"]}', flush=True)
    print(f'[DONE] {run_dir}', flush=True)
    return run_dir
