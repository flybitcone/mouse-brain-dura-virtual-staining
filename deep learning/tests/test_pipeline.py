import csv
import importlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import eval as evaluation
from dataset import subject_from_key
from utils.training import check_global_splits

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def fixture(root):
    """Only four real patches, retaining their actual split and mouse IDs."""
    for subset, count in [('train', 2), ('val', 1), ('test', 1)]:
        source = ROOT / 'dataset_patches' / 'BFtoIBA1' / subset
        with (source / 'metadata.csv').open(encoding='utf-8-sig') as f:
            rows = [next(csv.DictReader(f))] if count == 1 else None
        if rows is None:
            with (source / 'metadata.csv').open(encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                rows = [next(reader) for _ in range(count)]
        dest = root / 'BFtoIBA1' / subset
        for kind in ('input', 'target'):
            (dest / kind).mkdir(parents=True)
            for row in rows:
                # Use a small spatial fixture for most tests; pix2pix retains 256 below.
                arr = np.load(source / row[kind+'_path'], allow_pickle=False)
                np.save(dest / row[kind+'_path'], arr)
        with (dest / 'metadata.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def config(module, root, output):
    cfg = module.Config()
    cfg.data_root, cfg.output_root = str(root), str(output)
    cfg.epochs, cfg.batch_size, cfg.device, cfg.cpu_threads = 1, 1, 'cpu', 2
    cfg.max_train_batches = cfg.max_val_batches = cfg.max_test_batches = 1
    cfg.num_sample_images = 1
    if hasattr(cfg, 'features'):
        cfg.features = (2, 2, 4, 8, 16, 2)
    if hasattr(cfg, 'base_channels'):
        cfg.base_channels = 4
    if hasattr(cfg, 'n_residual_blocks'):
        cfg.n_residual_blocks = 1
    if hasattr(cfg, 'dim'):
        cfg.dim, cfg.num_blocks, cfg.num_refinement_blocks = 8, (1,1,1,1), 1
    if hasattr(cfg, 'embed_dim'):
        cfg.embed_dim, cfg.depths, cfg.num_heads = 12, (2,2), (3,3)
    return cfg


class PipelineTests(unittest.TestCase):
    def test_subject_parser_keeps_experiment_and_mouse(self):
        self.assertEqual(subject_from_key('EXP01_M001__LPS__FOV001_r0_c0'), 'EXP01_M001')
        self.assertEqual(subject_from_key('M006_Cortex_FOV001'), 'M006')

    def test_global_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            fixture(root)
            self.assertEqual(len(check_global_splits(root)), 3)
            metadata = root / 'BFtoIBA1' / 'test' / 'metadata.csv'
            text = metadata.read_text(encoding='utf-8-sig').replace('EXP01_M004', 'EXP01_M001')
            metadata.write_text(text, encoding='utf-8-sig')
            with self.assertRaisesRegex(ValueError, 'leakage'):
                check_global_splits(root)

    def test_all_eight_models_train_save_load_and_test(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            fixture(root / 'data')
            for name, module_name in evaluation.MODULES.items():
                with self.subTest(model=name):
                    module = importlib.import_module(module_name)
                    cfg = config(module, root/'data', root/'runs')
                    run = module.main(cfg)
                    checkpoint = torch.load(run/'checkpoints/best_model.pt', map_location='cpu', weights_only=True)
                    model, loaded_cfg = evaluation.load_model(name, checkpoint, torch.device('cpu'))
                    model.eval()
                    with torch.no_grad():
                        self.assertEqual(tuple(model(torch.zeros(1,3,256,256)).shape), (1,1,256,256))
                    result = json.loads((run/'test_metrics.json').read_text())
                    self.assertTrue(all(np.isfinite(v) for v in result['metrics'].values()))
                    self.assertEqual(result['samples'], 1)
                    self.assertTrue(result['limited_run'])
                    del checkpoint, model

    def test_resume_matches_uninterrupted_training_and_eval_cli(self):
        import Unet_train as module
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            fixture(root/'data')
            cfg = config(module, root/'data', root/'runs')
            first = module.main(cfg)
            cfg.resume, cfg.epochs = str(first/'checkpoints/last_model.pt'), 2
            resumed = module.main(cfg)
            cfg.resume = ''
            continuous = module.main(cfg)
            a = torch.load(resumed/'checkpoints/last_model.pt', weights_only=True, map_location='cpu')
            b = torch.load(continuous/'checkpoints/last_model.pt', weights_only=True, map_location='cpu')
            for key in a['model_state_dict']:
                self.assertTrue(torch.equal(a['model_state_dict'][key], b['model_state_dict'][key]), key)
            output = evaluation.main(['--model','unet','--checkpoint',str(resumed/'checkpoints/best_model.pt'),
                '--data-root',str(root/'data'),'--device','cpu','--cpu-threads','2','--save-max','1','--output',str(root/'eval')])
            self.assertTrue((output/'per_mouse.csv').is_file())
            self.assertEqual(len(list((output/'predictions').glob('*_pred.npy'))), 1)
            self.assertTrue(json.loads((output/'metrics.json').read_text())['provenance_verified'])
            metadata = root/'data/BFtoIBA1/train/metadata.csv'
            with metadata.open('a') as f:
                f.write('\n')
            cfg.resume, cfg.epochs = str(resumed/'checkpoints/last_model.pt'), 3
            with self.assertRaisesRegex(ValueError, 'metadata changed'):
                module.main(cfg)


if __name__ == '__main__':
    unittest.main(verbosity=2)
