"""Audit every NPY file, including files excluded from the active metadata index."""
import argparse
import csv
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np


def audit(root):
    report = {"subsets": [], "errors": [], "overlaps": []}
    global_keys = {s: set() for s in ('train', 'val', 'test')}
    for task in sorted(root.iterdir()):
        if not task.is_dir():
            continue
        keys = {}
        for subset in ("train", "val", "test"):
            directory = task / subset
            with (directory / 'metadata.csv').open(encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
            keys[subset] = set()
            for r in rows:
                subject = r.get('subject_id')
                if not subject:
                    match = re.search(r'(?:EXP\d+_)?M\d+', r['fov_key'])
                    if not match:
                        raise ValueError(f'Cannot identify subject: {r["fov_key"]}')
                    subject = match.group(0)
                keys[subset].add(subject)
            global_keys[subset].update(keys[subset])
            if len({r['patch_id'] for r in rows}) != len(rows):
                report['errors'].append(dict(file=str(directory / 'metadata.csv'), error='duplicate patch IDs'))
            entry = dict(task=task.name, subset=subset, indexed_pairs=len(rows), files=0,
                         excluded_files=0, minimum=1.0, maximum=0.0, shapes={})
            for kind in ('input', 'target'):
                indexed = {r[kind + '_path'] for r in rows}
                for file in sorted((directory / kind).glob('*.npy')):
                    entry['files'] += 1
                    if f'{kind}/{file.name}' not in indexed:
                        entry['excluded_files'] += 1
                    try:
                        arr = np.load(file, allow_pickle=False)
                        if arr.dtype != np.float32:
                            raise ValueError(f'unexpected dtype {arr.dtype}')
                        expected = (256, 256, 3) if kind == 'input' else (256, 256)
                        if arr.shape not in (expected, (256, 256, 1) if kind == 'target' else expected):
                            raise ValueError(f'unexpected shape {arr.shape}')
                        if not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 1:
                            raise ValueError('non-finite or outside [0, 1]')
                        entry['minimum'] = min(entry['minimum'], float(arr.min()))
                        entry['maximum'] = max(entry['maximum'], float(arr.max()))
                        shape = str(arr.shape) + ' ' + str(arr.dtype)
                        entry['shapes'][shape] = entry['shapes'].get(shape, 0) + 1
                    except Exception as exc:
                        report['errors'].append(dict(file=str(file), error=str(exc)))
                for relative in indexed:
                    if not (directory / relative).is_file():
                        report['errors'].append(dict(file=str(directory / relative), error='missing'))
            report['subsets'].append(entry)
            if entry['excluded_files']:
                report['errors'].append(dict(file=str(directory), error='unindexed files present'))
            print(json.dumps(entry), flush=True)
        for a, b in combinations(keys, 2):
            if keys[a] & keys[b]:
                report['overlaps'].append(dict(task=task.name, subsets=[a,b], subjects=sorted(keys[a]&keys[b])))
    for a, b in combinations(global_keys, 2):
        if global_keys[a] & global_keys[b]:
            report['overlaps'].append(dict(task='ALL_TASKS', subsets=[a,b], subjects=sorted(global_keys[a]&global_keys[b])))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).parent / 'dataset_patches')
    parser.add_argument('--output', type=Path, default=Path('dataset_audit.json'))
    args = parser.parse_args()
    report = audit(args.root)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    raise SystemExit(bool(report['errors'] or report['overlaps']))
