"""Restartable full-pixel verification in bounded batches; no sampling."""
import argparse
import csv
import json
from collections import defaultdict

import preprocess as p
import numpy as np

BATCH_SIZE = 2000
RESULTS = p.REPORT / 'validation_batches'


def read_rows(task, subset):
    directory = p.STAGE / 'dataset_patches' / task / subset
    metadata = directory / 'metadata.csv'
    with metadata.open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows or len({r['patch_id'] for r in rows}) != len(rows):
        raise ValueError('Empty subset or duplicate patch IDs')
    for kind in ('input', 'target'):
        if {r[kind + '_path'] for r in rows} != {f'{kind}/{x.name}' for x in (directory / kind).iterdir()}:
            raise ValueError('Index/file mismatch')
    return directory, metadata, rows


def verify(task, subset, start):
    directory, metadata, rows = read_rows(task, subset)
    if start < 0 or start >= len(rows) or start % BATCH_SIZE:
        raise ValueError('Invalid batch boundary')
    end = min(start + BATCH_SIZE, len(rows))
    current = None
    subjects = set()
    for row in rows[start:end]:
        if row['task'] != task or row['subset'] != subset:
            raise ValueError('Wrong task/subset')
        subjects.add(row['subject_id'])
        if current != row['fov_key']:
            current = row['fov_key']
            source_dir = p.STAGE / 'dataset_split' / task / subset
            ip = source_dir / 'input' / (current + '.tif')
            tp = source_dir / 'target' / (current + '.tif')
            if p.digest_file(ip) != row['input_sha256'] or p.digest_file(tp) != row['target_sha256']:
                raise ValueError('TIFF/source hash mismatch')
            x, _ = p.read_rgb(ip)
            y, _ = p.read_rgb(tp)
            x = x.astype(np.float32) / np.float32(255)
            y = y[..., int(row['target_channel'])].astype(np.float32) / np.float32(255)
        a = np.load(directory / row['input_path'], allow_pickle=False)
        b = np.load(directory / row['target_path'], allow_pickle=False)
        p.verify_patch(a, b)
        r, c = int(row['row']), int(row['col'])
        if not np.array_equal(a, x[r:r+256, c:c+256]) or not np.array_equal(b, y[r:r+256, c:c+256]):
            raise ValueError(f'Pixel mismatch: {row["patch_id"]}')
    p.write_json(RESULTS / f'{task}_{subset}_{start}.json', dict(
        task=task, subset=subset, start=start, end=end, checked=end-start,
        subjects=sorted(subjects), metadata_sha256=p.digest_file(metadata), status='passed'))
    print(f'Passed {task}/{subset} [{start}, {end})', flush=True)


def finish():
    subjects = defaultdict(set)
    checked = 0
    with (p.REPORT / 'pairs.csv').open(encoding='utf-8-sig') as f:
        pairs = list(csv.DictReader(f))
    with (p.REPORT / 'build_summary.csv').open(encoding='utf-8-sig') as f:
        summary = list(csv.DictReader(f))
    for task in sorted({r['task'] for r in pairs}):
        for subset in p.SUBSETS:
            _, metadata, rows = read_rows(task, subset)
            meta_hash = p.digest_file(metadata)
            # Also verify every expected source crop is indexed exactly once.
            expected = {(r['fov_key'], y, x) for r in pairs if r['task'] == task and r['subset'] == subset
                        for y in range(0, int(r['height'])//256*256, 256)
                        for x in range(0, int(r['width'])//256*256, 256)}
            actual = {(r['fov_key'], int(r['row']), int(r['col'])) for r in rows}
            if actual != expected or len(actual) != len(rows):
                raise ValueError('Source crop coverage mismatch')
            for start in range(0, len(rows), BATCH_SIZE):
                result = json.loads((RESULTS / f'{task}_{subset}_{start}.json').read_text(encoding='utf-8'))
                end = min(start+BATCH_SIZE, len(rows))
                if (result['status'], result['task'], result['subset'], result['start'], result['end'], result['checked'], result['metadata_sha256']) != ('passed', task, subset, start, end, end-start, meta_hash):
                    raise ValueError('Invalid or stale batch result')
                checked += result['checked']
                subjects[subset].update(result['subjects'])
    for i, a in enumerate(p.SUBSETS):
        for b in p.SUBSETS[i+1:]:
            if subjects[a] & subjects[b]:
                raise ValueError('Global subject leakage')
    if checked != sum(int(r['patches']) for r in summary):
        raise ValueError('Build summary count mismatch')
    p.write_json(p.REPORT / 'validation.json', dict(status='passed', checked_patch_pairs=checked,
        global_subject_overlap=[], source_copy_hashes_verified=True, every_patch_compared_to_source=True,
        source_crop_coverage_verified=True, method='exhaustive batches of 2000; no sampling',
        subjects={k: sorted(v) for k, v in subjects.items()}))
    with (p.REPORT / 'raw_inventory.csv').open(encoding='utf-8-sig') as f:
        raw_count = sum(1 for _ in csv.DictReader(f))
    p.write_json(p.STAGE / 'SUCCESS.json', dict(status='validated', summary=summary,
        raw_files=raw_count, source_inventory_sha256=p.digest_file(p.REPORT / 'raw_inventory.csv')))
    print(f'All {checked} patch pairs verified, complete crop coverage, no global subject overlap.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['BFto'+m for m in p.CHANNELS])
    parser.add_argument('--subset', choices=p.SUBSETS)
    parser.add_argument('--start', type=int)
    parser.add_argument('--finish', action='store_true')
    args = parser.parse_args()
    if args.finish:
        finish()
    elif args.task is None or args.subset is None or args.start is None:
        parser.error('--task, --subset and --start are required')
    else:
        verify(args.task, args.subset, args.start)
