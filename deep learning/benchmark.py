"""Run requested model/task/seed combinations under a shared training budget."""
import argparse
import csv
import importlib
import json
from itertools import product
from pathlib import Path
import numpy as np
from eval import MODULES
from utils.training import create_run_dir

TASKS = ('BFtoDAPI','BFtoIBA1','BFtoCD206','BFtoLYVE1','BFtoNeutrophil')


def aggregate(rows):
    results = []
    for model, task in sorted({(r['model'],r['task']) for r in rows}):
        group = [r for r in rows if (r['model'],r['task']) == (model,task)]
        row = dict(model=model, task=task, runs=len(group))
        for key in ('mae','nmse','psnr','ssim','mouse_mae','mouse_nmse','mouse_psnr','mouse_ssim'):
            values = [r[key] for r in group]
            row[key+'_mean'] = float(np.mean(values))
            row[key+'_seed_std'] = float(np.std(values, ddof=1)) if len(group)>1 else None
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODULES, default=list(MODULES))
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[42,43,44])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--data-root', default=str(Path(__file__).resolve().parent/'dataset_patches'))
    parser.add_argument('--output-root', default=str(Path(__file__).resolve().parent/'benchmark_runs'))
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('epochs must be positive and seeds must be unique')
    root = create_run_dir(args.output_root, 'batch', 'benchmark')
    (root/'plan.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    rows = []
    for model, task, seed in product(args.models, args.tasks, args.seeds):
        module = importlib.import_module(MODULES[model])
        cfg = module.Config()
        cfg.task, cfg.seed, cfg.epochs, cfg.schedule_epochs = task, seed, args.epochs, args.epochs
        cfg.warmup_epochs = min(5, args.epochs-1)
        cfg.batch_size, cfg.device, cfg.num_workers = args.batch_size, args.device, args.num_workers
        cfg.data_root, cfg.output_root = args.data_root, str(root/'runs')
        run = module.main(cfg)
        result = json.loads((run/'test_metrics.json').read_text(encoding='utf-8'))
        rows.append(dict(model=model, task=task, seed=seed, run=str(run), best_epoch=result['best_epoch'],
                         **result['metrics'], **{'mouse_'+k:v for k,v in result['macro_mouse'].items()}))
        with (root/'runs.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (root/'summary.json').write_text(json.dumps(aggregate(rows), indent=2), encoding='utf-8')
    print(f'Benchmark results: {root}')


if __name__ == '__main__':
    main()
