import tempfile
import unittest
import contextlib
import importlib.util
import io
from pathlib import Path
from unittest.mock import patch

import preprocess as p
import numpy as np
import tifffile


class PreprocessTests(unittest.TestCase):
    def test_full_context_prevents_fov_collision_and_legacy_marker_is_explicit(self):
        with tempfile.TemporaryDirectory(dir=p.ROOT) as tmp:
            root = Path(tmp)
            for context in ('LYVE1', 'LYVE1_CD206'):
                parent = root / 'raw_data' / 'EXP01_M001' / 'PBS' / 'Dura' / context
                for channel, suffix in [('Bright_Field', 'BF'), ('LYVE1', 'Neutrophil' if context == 'LYVE1_CD206' else 'IF')]:
                    path = parent / channel / f'FOV001_00X_{suffix}.tif'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'index-only fixture')
            with patch.object(p, 'ROOT', root), patch.object(p, 'RAW', root / 'raw_data'):
                inventory, pairs = p.discover()
            self.assertEqual(len({r['fov_key'] for r in pairs}), 2)
            self.assertEqual({r['task'] for r in pairs}, {'BFtoLYVE1'})
            self.assertEqual(sum(r['legacy_suffix_correction'] for r in inventory), 1)

    def test_rgba_discards_only_opaque_alpha(self):
        with tempfile.TemporaryDirectory(dir=p.ROOT) as tmp:
            path = Path(tmp) / 'test.tif'
            arr = np.zeros((256, 256, 4), dtype=np.uint8)
            arr[..., 0], arr[..., 3] = 127, 255
            tifffile.imwrite(path, arr, photometric='rgb')
            rgb, count = p.read_rgb(path)
            self.assertEqual(count, 4)
            np.testing.assert_array_equal(rgb, arr[..., :3])
            arr[0, 0, 3] = 0
            tifffile.imwrite(path, arr, photometric='rgb')
            with self.assertRaisesRegex(ValueError, 'Non-opaque'):
                p.read_rgb(path)

    def test_actual_metadata_split_is_global_and_every_task_has_three_subsets(self):
        if not p.RAW.is_dir() or not any(p.RAW.rglob('*.tif')):
            self.skipTest('Download and extract the raw dataset to raw_data/ for this integration test.')
        _, pairs = p.discover()
        mapping, info = p.choose_split(pairs)
        self.assertEqual([len(info['subjects'][s]) for s in p.SUBSETS], [13, 3, 3])
        self.assertEqual(len(mapping), 19)
        for task in {r['task'] for r in pairs}:
            self.assertEqual({mapping[r['subject_id']] for r in pairs if r['task'] == task}, set(p.SUBSETS))

    def test_fixed_normalization_preserves_dim_signal(self):
        inp = np.full((256, 256, 3), 51, dtype=np.uint8).astype(np.float32) / np.float32(255)
        target = np.zeros((256, 256), dtype=np.float32)
        target[0, 0] = np.float32(25) / np.float32(255)
        p.verify_patch(inp, target)
        self.assertAlmostEqual(float(inp[0, 0, 0]), 0.2)
        self.assertLess(float(target.max()), 0.1)
        target[0, 0] = np.nan
        with self.assertRaises(ValueError):
            p.verify_patch(inp, target)

    def test_auditor_detects_cross_task_mouse_leakage(self):
        spec = importlib.util.spec_from_file_location('audit_dataset', p.ROOT / 'deep learning' / 'audit_dataset.py')
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        with tempfile.TemporaryDirectory(dir=p.ROOT) as tmp:
            root = Path(tmp)
            for task, mice in [('BFtoDAPI', [1, 2, 3]), ('BFtoIBA1', [4, 5, 1])]:
                for subset, mouse in zip(p.SUBSETS, mice):
                    directory = root / task / subset
                    row = dict(subject_id=f'EXP01_M{mouse:03}', fov_key=f'EXP01_M{mouse:03}__FOV001',
                               patch_id='patch', input_path='input/patch.npy', target_path='target/patch.npy')
                    for kind, shape in [('input', (256, 256, 3)), ('target', (256, 256))]:
                        (directory / kind).mkdir(parents=True)
                        np.save(directory / row[kind + '_path'], np.zeros(shape, dtype=np.float32))
                    p.write_csv(directory / 'metadata.csv', [row])
            with contextlib.redirect_stdout(io.StringIO()):
                report = audit.audit(root)
            self.assertEqual(report['errors'], [])
            self.assertEqual(report['overlaps'], [dict(task='ALL_TASKS', subsets=['train', 'test'], subjects=['EXP01_M001'])])


if __name__ == '__main__':
    unittest.main()
