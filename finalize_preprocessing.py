"""Verify publication coverage and refresh the training-side audit report."""
import csv
import json

import preprocess as p


def finalize():
    validation = json.loads((p.REPORT / 'validation.json').read_text(encoding='utf-8'))
    success = json.loads((p.REPORT / 'SUCCESS.json').read_text(encoding='utf-8'))
    if validation['status'] != 'passed' or success['status'] != 'validated':
        raise ValueError('Full pixel verification has not passed')
    if success['source_inventory_sha256'] != p.digest_file(p.REPORT / 'raw_inventory.csv'):
        raise ValueError('Source inventory changed since validation')
    with (p.REPORT / 'pairs.csv').open(encoding='utf-8-sig') as f:
        pairs = list(csv.DictReader(f))
    expected = {(r['task'], r['subset'], r['fov_key']):
                {(y, x) for y in range(0, int(r['height'])//256*256, 256)
                 for x in range(0, int(r['width'])//256*256, 256)} for r in pairs}
    if len(expected) != len(pairs):
        raise ValueError('Duplicate source pair key')
    entries = []
    total = 0
    for task in sorted({r['task'] for r in pairs}):
        for subset in p.SUBSETS:
            directory = p.ROOT / 'deep learning' / 'dataset_patches' / task / subset
            with (directory / 'metadata.csv').open(encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
            coords = {}
            for row in rows:
                key = (task, subset, row['fov_key'])
                coords.setdefault(key, set()).add((int(row['row']), int(row['col'])))
            if sum(map(len, coords.values())) != len(rows):
                raise ValueError('Duplicate crop coordinate')
            subset_expected = {k: v for k, v in expected.items() if k[:2] == (task, subset)}
            if coords != subset_expected:
                raise ValueError(f'Missing/extra source crops in {task}/{subset}')
            for kind in ('input', 'target'):
                paths = {directory / r[kind + '_path'] for r in rows}
                if paths != set((directory / kind).glob('*.npy')):
                    raise ValueError('Published patch file/index mismatch')
            split_dir = p.ROOT / 'dataset_split' / task / subset
            with (split_dir / 'metadata.csv').open(encoding='utf-8-sig') as f:
                fovs = list(csv.DictReader(f))
            if {r['fov_key'] for r in fovs} != {k[2] for k in subset_expected} or len(fovs) != len(subset_expected):
                raise ValueError('Published TIFF manifest coverage mismatch')
            for kind in ('input', 'target'):
                if {split_dir / r[kind + '_path'] for r in fovs} != set((split_dir / kind).glob('*.tif')):
                    raise ValueError('Published TIFF file/index mismatch')
            entries.append(dict(task=task, subset=subset, indexed_pairs=len(rows),
                files=2*len(rows), excluded_files=0, fovs=len(fovs),
                subjects=sorted({r['subject_id'] for r in rows}),
                shapes={'(256, 256, 3) float32': len(rows), '(256, 256) float32': len(rows)},
                verified_value_bounds=[0, 1]))
            total += len(rows)
    if total != validation.get('indexed_patch_pairs', validation.get('checked_patch_pairs')):
        raise ValueError('Published count differs from full pixel verification')
    report = dict(status='passed', subsets=entries, errors=[], overlaps=[],
        indexed_patch_pairs=total, original_files=success['raw_files'],
        verification=validation.get('scope', 'full pixel validation'),
        every_patch_compared_to_source=validation.get('every_patch_compared_to_source', False),
        full_validation_report='preprocessing_reports/validation.json',
        raw_inventory_sha256=p.digest_file(p.REPORT / 'raw_inventory.csv'))
    p.write_json(p.ROOT / 'deep learning' / 'dataset_audit.json', report)
    p.write_json(p.REPORT / 'publication_validation.json', report)
    p.write_json(p.REPORT / 'code_hashes.json', {name: p.digest_file(p.ROOT / name)
        for name in ('preprocess.py', 'validate_preprocessing_leakage.py', 'finalize_preprocessing.py', 'Publish-PreprocessedData.ps1',
                     'DataPreprocess.ipynb', 'deep learning/audit_dataset.py')})
    print(f'Publication verified: {len(pairs)} FOV pairs, {total} patch pairs, no extra or missing files.')


if __name__ == '__main__':
    finalize()
