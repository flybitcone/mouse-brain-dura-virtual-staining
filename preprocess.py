"""Reproducible, mouse-disjoint virtual-staining data preparation.

python preprocess.py --inspect
python preprocess.py --build
python preprocess.py --validate  # resume verification of a completed staging build
Writes a staging directory. Promote it only after validate_build succeeds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / '.preprocess_deps').exists():
    sys.path.insert(0, str(ROOT / '.preprocess_deps'))
import numpy as np
import tifffile

RAW = ROOT / 'raw_data'
REPORT = ROOT / 'preprocessing_reports'
STAGE = ROOT / 'preprocessing_build'
CHANNELS = {'DAPI': 2, 'IBA1': 0, 'CD206': 1, 'LYVE1': 0, 'Neutrophil': 0}
SUBSETS = ('train', 'val', 'test')
PATCH_SIZE = 256
SEED = 42
RATIOS = np.array([0.70, 0.15, 0.15])


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rgb(path):
    # Read the actual single page, not unreliable embedded OME dimensions.
    with tifffile.TiffFile(path) as tif:
        if len(tif.pages) != 1:
            raise ValueError(f'Multi-page TIFF needs an explicit policy: {path}')
        arr = tif.pages[0].asarray()
    if arr.dtype != np.uint8:
        raise ValueError(f'Expected audited uint8 TIFF: {path}: {arr.dtype}')
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f'Expected RGB/RGBA: {path}: {arr.shape}')
    if arr.shape[-1] == 4 and not np.all(arr[..., 3] == 255):
        raise ValueError(f'Non-opaque alpha needs explicit treatment: {path}')
    return np.ascontiguousarray(arr[..., :3]), arr.shape[-1]


def discover():
    """Match full context + FOV + magnification; fail on missing/ambiguous data."""
    groups = defaultdict(dict)
    inventory = []
    for path in sorted(RAW.rglob('*')):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ('.tif', '.tiff'):
            raise ValueError(f'Unrecognized raw file: {path}')
        rel = path.relative_to(RAW)
        subject, condition, tissue = rel.parts[:3]
        if not re.fullmatch(r'EXP\d+_M\d+', subject):
            raise ValueError(f'Invalid subject ID: {subject}')
        marker = path.parent.name
        if marker not in {'Bright_Field', *CHANNELS}:
            raise ValueError(f'Unknown channel: {path}')
        match = re.fullmatch(r'(FOV\d+)_([0-9]+X)_(BF|DAPI|IF|IBA1|CD206|LYVE1|Neutrophil)', path.stem)
        if match is None:
            raise ValueError(f'Unrecognized filename: {path}')
        fov, magnification, suffix = match.groups()
        allowed = {'BF'} if marker == 'Bright_Field' else {marker, 'IF'}
        # Legacy notebook cell 4 saved LYVE1 red-channel files with this suffix.
        # Restrict the exception to that exact panel; retain the original path.
        legacy_suffix = marker == 'LYVE1' and path.parent.parent.name == 'LYVE1_CD206' and suffix == 'Neutrophil'
        if legacy_suffix:
            allowed.add('Neutrophil')
        if suffix not in allowed:
            raise ValueError(f'Channel/suffix mismatch: {path}')
        context = path.parent.parent.relative_to(RAW).as_posix()
        key = context + '/' + fov + '_' + magnification
        if marker in groups[key]:
            raise ValueError(f'Duplicate FOV/channel key: {key}/{marker}')
        row = dict(path=path.relative_to(ROOT).as_posix(), subject_id=subject,
                   mouse_id=subject.split('_')[-1], condition=condition, tissue=tissue,
                   context=context, channel=marker, fov=fov, magnification=magnification,
                   fov_key=key.replace('/', '__'), bytes=path.stat().st_size,
                   legacy_suffix_correction=bool(legacy_suffix))
        inventory.append(row)
        groups[key][marker] = row
    pairs = []
    for key, channels in sorted(groups.items()):
        if 'Bright_Field' not in channels:
            raise ValueError(f'Missing BF: {key}')
        if len(channels) == 1:
            raise ValueError(f'BF with no target: {key}')
        inp = channels['Bright_Field']
        for marker in CHANNELS:
            if marker in channels:
                target = channels[marker]
                pairs.append(dict(task='BFto' + marker, subject_id=inp['subject_id'],
                                  mouse_id=inp['mouse_id'], condition=inp['condition'],
                                  tissue=inp['tissue'], context=inp['context'],
                                  fov_key=inp['fov_key'], magnification=inp['magnification'],
                                  source_input=inp['path'], source_target=target['path'],
                                  target_channel=CHANNELS[marker]))
    return inventory, pairs


def choose_split(pairs):
    """Fixed 13/3/3 global split balanced using metadata counts only."""
    subjects = sorted({p['subject_id'] for p in pairs})
    n = len(subjects)
    n_val = max(1, round(n * RATIOS[1]))
    n_test = max(1, round(n * RATIOS[2]))
    n_train = n - n_val - n_test
    tasks = sorted({p['task'] for p in pairs})
    strata = sorted({(p['task'], p['condition']) for p in pairs})
    matrix = np.zeros((n, len(tasks) + len(strata)), dtype=np.float64)
    for p in pairs:
        i = subjects.index(p['subject_id'])
        matrix[i, tasks.index(p['task'])] += 1
        matrix[i, len(tasks) + strata.index((p['task'], p['condition']))] += 1
    rng = np.random.default_rng(SEED)
    best = None
    # Search a deterministic, fixed number of metadata-only assignments.
    for _ in range(30000):
        order = rng.permutation(n)
        partitions = (order[:n_train], order[n_train:n_train+n_val], order[n_train+n_val:])
        counts = np.stack([matrix[ix].sum(0) for ix in partitions])
        if (counts[:, :len(tasks)] == 0).any():
            continue
        fractions = counts / matrix.sum(0)
        score = float(np.square(fractions[:, :len(tasks)] - RATIOS[:, None]).sum()
                      + 0.25 * np.square(fractions[:, len(tasks):] - RATIOS[:, None]).sum())
        if best is None or score < best[0]:
            best = score, partitions
    if best is None:
        raise ValueError('Cannot populate every task in all three subsets')
    mapping = {subjects[int(i)]: subset for subset, ix in zip(SUBSETS, best[1]) for i in ix}
    return mapping, dict(seed=SEED, requested_ratios=RATIOS.tolist(), candidates=30000,
                         objective=best[0], method='metadata-only task/condition FOV-count balancing',
                         subjects={s: sorted(k for k, v in mapping.items() if v == s) for s in SUBSETS})


def inspect():
    inventory, pairs = discover()
    mapping, split_info = choose_split(pairs)
    for p in pairs:
        p['subset'] = mapping[p['subject_id']]
    write_csv(REPORT / 'discovery_inventory.csv', inventory)
    write_csv(REPORT / 'discovered_pairs.csv', pairs)
    write_json(REPORT / 'split.json', split_info)
    summary = []
    for task in sorted({p['task'] for p in pairs}):
        for subset in SUBSETS:
            rows = [p for p in pairs if p['task'] == task and p['subset'] == subset]
            summary.append(dict(task=task, subset=subset, fovs=len(rows),
                                subjects=','.join(sorted({p['subject_id'] for p in rows}))))
    write_csv(REPORT / 'split_summary.csv', summary)
    print(json.dumps(dict(raw_files=len(inventory), pairs=len(pairs), split=split_info,
                          summary=summary), ensure_ascii=False, indent=2), flush=True)
    return inventory, pairs


def audit_raw(inventory, pairs):
    by_path = {}
    duplicate_groups = defaultdict(list)
    for i, row in enumerate(inventory):
        path = ROOT / row['path']
        rgb, channels = read_rgb(path)
        row.update(height=rgb.shape[0], width=rgb.shape[1], stored_channels=channels,
                   dtype='uint8', sha256=digest_file(path),
                   rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                   mean_r=float(rgb[..., 0].mean()), mean_g=float(rgb[..., 1].mean()),
                   mean_b=float(rgb[..., 2].mean()))
        duplicate_groups[(row['channel'], row['rgb_sha256'])].append(row)
        by_path[row['path']] = row
        if (i + 1) % 100 == 0:
            print(f'Raw audit {i+1}/{len(inventory)}', flush=True)
    write_csv(REPORT / 'raw_inventory.csv', inventory)
    duplicate_report = []
    for rows in duplicate_groups.values():
        if len(rows) > 1:
            duplicate_report.append(dict(channel=rows[0]['channel'],
                paths=[r['path'] for r in rows],
                subjects=sorted({r['subject_id'] for r in rows})))
    write_json(REPORT / 'duplicates.json', duplicate_report)
    subject_split = {p['subject_id']: p['subset'] for p in pairs}
    leaking = [d for d in duplicate_report if len({subject_split[s] for s in d['subjects']}) > 1]
    if leaking:
        raise ValueError(f'Exact image duplicates across splits: see duplicates.json ({len(leaking)})')
    for p in pairs:
        a, b = by_path[p['source_input']], by_path[p['source_target']]
        if (a['height'], a['width']) != (b['height'], b['width']):
            raise ValueError(f'Paired dimensions differ, alignment review required: {p["fov_key"]}')
        if min(a['height'], a['width']) < PATCH_SIZE:
            raise ValueError(f'Image too small: {p["fov_key"]}')
        p.update(height=a['height'], width=a['width'],
                 input_sha256=a['sha256'], target_sha256=b['sha256'])
    write_csv(REPORT / 'pairs.csv', pairs)
    return by_path


def verify_patch(inp, target):
    for arr, shape in ((inp, (PATCH_SIZE, PATCH_SIZE, 3)), (target, (PATCH_SIZE, PATCH_SIZE))):
        if arr.shape != shape or arr.dtype != np.float32:
            raise ValueError(f'Invalid patch: {arr.shape}, {arr.dtype}')
        if not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 1:
            raise ValueError('Invalid patch values')


def build():
    if STAGE.exists():
        raise FileExistsError(f'Staging directory already exists: {STAGE}; inspect before removing')
    inventory, pairs = inspect()
    by_path = audit_raw(inventory, pairs)
    required = sum(by_path[p['source_input']]['bytes'] + by_path[p['source_target']]['bytes']
                   + (p['height']//PATCH_SIZE)*(p['width']//PATCH_SIZE)*(PATCH_SIZE**2*16+512)
                   for p in pairs)
    free = shutil.disk_usage(ROOT).free
    if required * 1.08 > free:
        raise OSError(f'Insufficient staging disk space: need about {required/1e9:.1f} GB, free {free/1e9:.1f} GB')
    print(f'Building {len(pairs)} pairs, estimated {required/1e9:.1f} GB', flush=True)
    STAGE.mkdir()
    summary = []
    for task in sorted({p['task'] for p in pairs}):
        for subset in SUBSETS:
            selected = [p for p in pairs if p['task'] == task and p['subset'] == subset]
            split_dir = STAGE / 'dataset_split' / task / subset
            patch_dir = STAGE / 'dataset_patches' / task / subset
            for base in (split_dir, patch_dir):
                for kind in ('input', 'target'):
                    (base / kind).mkdir(parents=True)
            meta_rows, fov_rows = [], []
            for i, p in enumerate(selected):
                inp, _ = read_rgb(ROOT / p['source_input'])
                target_rgb, _ = read_rgb(ROOT / p['source_target'])
                inp = inp.astype(np.float32) / np.float32(255)
                target = target_rgb[..., p['target_channel']].astype(np.float32) / np.float32(255)
                h, w = inp.shape[:2]
                hc, wc = h//PATCH_SIZE*PATCH_SIZE, w//PATCH_SIZE*PATCH_SIZE
                base = dict(p, normalization='uint8_div_255', normalization_offset=0,
                            normalization_scale=255, retained_height=hc, retained_width=wc,
                            discarded_bottom=h-hc, discarded_right=w-wc)
                fov_row = dict(base)
                for kind in ('input', 'target'):
                    relative = f'{kind}/{p["fov_key"]}.tif'
                    shutil.copy2(ROOT / p['source_' + kind], split_dir / relative)
                    fov_row[kind + '_path'] = relative
                fov_rows.append(fov_row)
                for r in range(0, hc, PATCH_SIZE):
                    for c in range(0, wc, PATCH_SIZE):
                        patch_id = f'{p["fov_key"]}_r{r}_c{c}'
                        ip, tp = inp[r:r+PATCH_SIZE, c:c+PATCH_SIZE], target[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
                        verify_patch(ip, tp)
                        row = dict(base, patch_id=patch_id, row=r, col=c,
                                   input_path=f'input/{patch_id}.npy', target_path=f'target/{patch_id}.npy')
                        np.save(patch_dir / row['input_path'], ip, allow_pickle=False)
                        np.save(patch_dir / row['target_path'], tp, allow_pickle=False)
                        meta_rows.append(row)
                if (i+1) % 50 == 0:
                    print(f'{task}/{subset}: {i+1}/{len(selected)} FOVs', flush=True)
            write_csv(split_dir / 'metadata.csv', fov_rows)
            write_csv(patch_dir / 'metadata.csv', meta_rows)
            summary.append(dict(task=task, subset=subset, mice=len({p['subject_id'] for p in selected}),
                                fovs=len(selected), patches=len(meta_rows)))
            print(summary[-1], flush=True)
    write_csv(REPORT / 'build_summary.csv', summary)
    from validate_preprocessing_leakage import validate
    validate()


def validate_build(patch_root, split_root):
    """Read back every saved patch and compare against the copied source TIFF."""
    subjects = defaultdict(set)
    checked = 0
    for task in sorted(patch_root.iterdir()):
        if not task.is_dir():
            continue
        for subset in SUBSETS:
            directory = task / subset
            with (directory / 'metadata.csv').open(encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
            if not rows:
                raise ValueError(f'Empty subset: {directory}')
            ids = [r['patch_id'] for r in rows]
            if len(ids) != len(set(ids)):
                raise ValueError(f'Duplicate patch IDs: {directory}')
            for kind in ('input', 'target'):
                actual = {p.name for p in (directory / kind).iterdir()}
                expected = {Path(r[kind + '_path']).name for r in rows}
                if actual != expected:
                    raise ValueError(f'Unindexed/missing files in {directory}/{kind}')
            current_fov = None
            for row_index, row in enumerate(rows, 1):
                subjects[subset].add(row['subject_id'])
                if row['task'] != task.name or row['subset'] != subset:
                    raise ValueError('Wrong task/subset metadata')
                if current_fov != row['fov_key']:
                    current_fov = row['fov_key']
                    raw_dir = split_root / task.name / subset
                    input_tif = raw_dir / 'input' / (current_fov + '.tif')
                    target_tif = raw_dir / 'target' / (current_fov + '.tif')
                    if digest_file(input_tif) != row['input_sha256'] or digest_file(target_tif) != row['target_sha256']:
                        raise ValueError('Source copy hash mismatch')
                    src_in, _ = read_rgb(input_tif)
                    src_tg, _ = read_rgb(target_tif)
                    src_in = src_in.astype(np.float32) / np.float32(255)
                    src_tg = src_tg[..., int(row['target_channel'])].astype(np.float32) / np.float32(255)
                inp = np.load(directory / row['input_path'], allow_pickle=False)
                target = np.load(directory / row['target_path'], allow_pickle=False)
                verify_patch(inp, target)
                r, c = int(row['row']), int(row['col'])
                if not np.array_equal(inp, src_in[r:r+PATCH_SIZE, c:c+PATCH_SIZE]) or not np.array_equal(target, src_tg[r:r+PATCH_SIZE, c:c+PATCH_SIZE]):
                    raise ValueError(f'Patch/source mismatch: {row["patch_id"]}')
                checked += 1
                if row_index % 1000 == 0:
                    print(f'Verifying {task.name}/{subset}: {row_index}/{len(rows)}', flush=True)
            print(f'Validated {task.name}/{subset}: {len(rows)} patches', flush=True)
    for i, a in enumerate(SUBSETS):
        for b in SUBSETS[i+1:]:
            if subjects[a] & subjects[b]:
                raise ValueError(f'Global subject leakage: {a}/{b}')
    write_json(REPORT / 'validation.json', dict(status='passed', checked_patch_pairs=checked,
               global_subject_overlap=[], source_copy_hashes_verified=True,
               every_patch_compared_to_source=True,
               subjects={k: sorted(v) for k, v in subjects.items()}))


def validate_existing():
    with (REPORT / 'build_summary.csv').open(encoding='utf-8-sig') as f:
        summary = list(csv.DictReader(f))
    with (REPORT / 'pairs.csv').open(encoding='utf-8-sig') as f:
        pairs = list(csv.DictReader(f))
    expected = Counter()
    for row in pairs:
        expected[(row['task'], row['subset'])] += (int(row['height'])//PATCH_SIZE)*(int(row['width'])//PATCH_SIZE)
    actual = {(r['task'], r['subset']): int(r['patches']) for r in summary}
    if dict(expected) != actual or len(summary) != len(CHANNELS)*len(SUBSETS):
        raise ValueError('Staged build summary does not cover all source pairs')
    validate_build(STAGE / 'dataset_patches', STAGE / 'dataset_split')
    validation = json.loads((REPORT / 'validation.json').read_text(encoding='utf-8'))
    if validation['checked_patch_pairs'] != sum(expected.values()):
        raise ValueError('Verified patch count differs from source dimensions')
    with (REPORT / 'raw_inventory.csv').open(encoding='utf-8-sig') as f:
        raw_count = sum(1 for _ in csv.DictReader(f))
    write_json(STAGE / 'SUCCESS.json', dict(status='validated', summary=summary,
               raw_files=raw_count, source_inventory_sha256=digest_file(REPORT / 'raw_inventory.csv')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--inspect', action='store_true')
    mode.add_argument('--build', action='store_true')
    mode.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    if args.inspect:
        inspect()
    elif args.validate:
        validate_existing()
    else:
        build()
