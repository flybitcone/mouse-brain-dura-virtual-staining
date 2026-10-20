import importlib
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import VirtualStainingPatchDataset
from utils.metrics import compute_all_metrics, compute_nmse
from utils.training import check_datasets, create_run_dir, validate_config


torch.set_num_threads(2)


class TrainingRegressionTests(unittest.TestCase):
    def test_metrics_are_independent_of_batch_partition(self):
        torch.manual_seed(42)
        target = torch.rand(3, 1, 16, 16)
        pred = target * torch.tensor([0.1, 0.5, 0.9]).view(3, 1, 1, 1)
        together = compute_all_metrics(pred, target)
        separate = [compute_all_metrics(p[None], t[None]) for p, t in zip(pred, target)]
        for key in together:
            self.assertAlmostEqual(together[key], sum(m[key] for m in separate) / 3, places=5)
        self.assertGreater(compute_nmse(torch.ones_like(target), torch.zeros_like(target)), 0)
        self.assertEqual(compute_nmse(torch.zeros_like(target), torch.zeros_like(target)), 0)

    def test_invalid_targets_not_silently_clipped(self):
        with self.assertRaises(ValueError):
            compute_all_metrics(torch.zeros(1, 1, 16, 16), torch.full((1, 1, 16, 16), 2.0))

    def test_data_errors_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            subset = Path(tmp) / 'task/train'
            (subset / 'input').mkdir(parents=True)
            (subset / 'target').mkdir()
            np.save(subset / 'input/a.npy', np.zeros((16, 16, 3), np.float32))
            with self.assertRaises(ValueError):
                VirtualStainingPatchDataset(tmp, 'task', 'train')
            np.save(subset / 'target/a.npy', np.zeros((16, 16, 2), np.float32))
            ds = VirtualStainingPatchDataset(tmp, 'task', 'train')
            with self.assertRaises(ValueError):
                ds[0]

    def test_new_runs_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = create_run_dir(tmp, 'task', 'model')
            (a / 'keep.txt').write_text('original')
            b = create_run_dir(tmp, 'task', 'model')
            self.assertNotEqual(a, b)
            self.assertEqual((a / 'keep.txt').read_text(), 'original')

    def test_all_real_dataset_splits(self):
        from Unet_train import Config
        cfg = Config()
        for directory in Path(cfg.data_root).iterdir():
            if directory.is_dir():
                datasets = [VirtualStainingPatchDataset(cfg.data_root, directory.name, s,
                            return_metadata=False) for s in ('train', 'val', 'test')]
                check_datasets(cfg, *datasets)

    def test_supervised_models_train_and_validate(self):
        for name in ('Unet_train', 'ResUnet_train', 'AttUnet_train'):
            with self.subTest(name=name):
                module = importlib.import_module(name)
                cfg = module.Config()
                if hasattr(cfg, 'features'):
                    cfg.features = (2, 2, 4, 8, 16, 2)
                else:
                    cfg.base_channels = 2
                model = module.build_model(cfg)
                loader = DataLoader([(torch.rand(3, 32, 32), torch.rand(1, 32, 32))])
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
                scaler = torch.amp.GradScaler('cuda', enabled=False)
                train = module.train_one_epoch(model, loader, optimizer, module.CombinedLoss(),
                                               torch.device('cpu'), scaler, use_amp=False)
                val = module.validate_one_epoch(model, loader, module.CombinedLoss(),
                                                torch.device('cpu'), use_amp=False)
                self.assertTrue(all(math.isfinite(v) for v in [*train.values(), *val.values()]))
                self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_pix2pix_real_architecture_step(self):
        import Pix2pixcGAN_train as module
        generator, discriminator = module.GeneratorUNet(), module.Discriminator()
        loader = DataLoader([(torch.rand(3, 256, 256), torch.rand(1, 256, 256))])
        opt_g = torch.optim.SGD(generator.parameters(), lr=1e-4)
        opt_d = torch.optim.SGD(discriminator.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        result = module.train_one_epoch(generator, discriminator, loader, opt_g, opt_d,
                                        module.Pix2PixLoss(), torch.device('cpu'), scaler, scaler,
                                        use_amp=False)
        self.assertTrue(all(math.isfinite(v) for v in result.values()))
        self.assertTrue(all(p.requires_grad for p in discriminator.parameters()))

    def test_cyclegan_unequal_channels_train_and_validate(self):
        import CycleGAN_train as module
        cfg = module.Config()
        validate_config(cfg)
        cfg.lambda_identity = 5
        with self.assertRaises(ValueError):
            validate_config(cfg)
        ga = module.ResNetGenerator(3, 1, n_residual_blocks=1)
        gb = module.ResNetGenerator(1, 3, n_residual_blocks=1)
        da, db = module.Discriminator(3), module.Discriminator(1)
        loader = DataLoader([dict(A=torch.rand(3, 32, 32), B=torch.rand(1, 32, 32),
                                 B_pair=torch.rand(1, 32, 32))])
        opt_g = torch.optim.SGD(list(ga.parameters()) + list(gb.parameters()), lr=1e-4)
        opt_a = torch.optim.SGD(da.parameters(), lr=1e-4)
        opt_b = torch.optim.SGD(db.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        criterion = module.CycleGANLoss()
        result = module.train_one_epoch(ga, gb, da, db, loader, opt_g, opt_a, opt_b,
            criterion, module.ImagePool(2), module.ImagePool(2), torch.device('cpu'),
            scaler, scaler, scaler, use_amp=False)
        val = module.validate_one_epoch(ga, gb, da, db, loader, criterion,
                                       torch.device('cpu'), use_amp=False)
        self.assertEqual(result['loss_identity'], 0.0)
        self.assertTrue(all(math.isfinite(v) for v in [*result.values(), *val.values()]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
