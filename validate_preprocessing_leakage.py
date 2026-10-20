"""Audit global mouse/FOV/exact-image leakage without rereading image pixels."""
import csv
import json
from collections import defaultdict
import preprocess as p


def validate():
    with (p.REPORT / 'raw_inventory.csv').open(encoding='utf-8-sig') as f:
        inventory = list(csv.DictReader(f))
    with (p.REPORT / 'build_summary.csv').open(encoding='utf-8-sig') as f:
        summary = list(csv.DictReader(f))
    split = json.loads((p.REPORT / 'split.json').read_text(encoding='utf-8'))['subjects']
    assignments = {}
    for subset, mice in split.items():
        for mouse in mice:
            if mouse in assignments:
                raise ValueError(f'Mouse leakage: {mouse}')
            assignments[mouse] = subset
    if set(assignments) != {r['subject_id'] for r in inventory}:
        raise ValueError('Subject inventory/split mismatch')
    source_subject = {r['path']: r['subject_id'] for r in inventory}
    hash_subsets = defaultdict(set)
    for row in inventory:
        hash_subsets[row['rgb_sha256']].add(assignments[row['subject_id']])
    if any(len(s) > 1 for s in hash_subsets.values()):
        raise ValueError('Exact source RGB duplicates across subsets')
    subjects = defaultdict(set)
    fov_subsets = defaultdict(set)
    indexed = 0
    for entry in summary:
        task, subset = entry['task'], entry['subset']
        directory = p.STAGE / 'dataset_patches' / task / subset
        with (directory / 'metadata.csv').open(encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != int(entry['patches']):
            raise ValueError('Build count mismatch')
        for row in rows:
            mouse = row['subject_id']
            if (assignments[mouse] != subset or row['subset'] != subset or row['task'] != task
                or source_subject[row['source_input']] != mouse or source_subject[row['source_target']] != mouse
                or not row['fov_key'].startswith(mouse + '__')):
                raise ValueError(f'Inconsistent subject/source assignment: {row["patch_id"]}')
            subjects[subset].add(mouse)
            fov_subsets[row['fov_key']].add(subset)
        indexed += len(rows)
    if any(len(s) > 1 for s in fov_subsets.values()):
        raise ValueError('FOV leakage across tasks/subsets')
    report = dict(status='passed', scope='global mouse, FOV and exact source RGB duplicate leakage',
        indexed_patch_pairs=indexed, global_subject_overlap=[], global_fov_overlap=[],
        cross_subset_exact_image_duplicates=0, original_files=len(inventory),
        every_patch_compared_to_source=False,
        full_pixel_validation='Stopped at user request; generation-time shape/range checks completed.',
        subjects={k: sorted(v) for k, v in subjects.items()}, subsets=summary, errors=[], overlaps=[])
    p.write_json(p.REPORT / 'validation.json', report)
    p.write_json(p.STAGE / 'SUCCESS.json', dict(status='validated', scope=report['scope'], summary=summary,
        raw_files=len(inventory), source_inventory_sha256=p.digest_file(p.REPORT / 'raw_inventory.csv')))
    print(f'Leakage audit passed: {len(assignments)} mice, {indexed} indexed patch pairs; zero mouse/FOV/exact-image overlap.')


if __name__ == '__main__':
    validate()
