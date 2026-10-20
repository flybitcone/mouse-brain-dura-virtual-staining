#%%
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def subject_from_key(key: str) -> str:
    match = re.search(r'(?:EXP\d+_)?M\d+(?=_|$)', key)
    if not match:
        raise ValueError(f'Cannot identify mouse from key: {key}')
    return match.group(0)


class VirtualStainingPatchDataset(Dataset):
    """
    适用于 patch 级数据集的 PyTorch Dataset。

    期望目录结构:
        root/
        └── BFtoXXX/
            ├── train/
            │   ├── input/
            │   ├── target/
            │   └── metadata.csv
            ├── val/
            └── test/

    每个 input/target 文件为 .npy
    - input:  H x W x C 或 H x W
    - target: H x W 或 H x W x 1

    输出:
        {
            "input": Tensor[C, H, W],
            "target": Tensor[1, H, W],
            "input_path": str,
            "target_path": str,
            "patch_id": str,
            "fov_key": str,
            "task": str,
            "subset": str,
        }
    """

    def __init__(
        self,
        root: str | Path,
        task: str,
        subset: str,
        input_transform: Optional[Callable[[Tensor], Tensor]] = None,
        target_transform: Optional[Callable[[Tensor], Tensor]] = None,
        joint_transform: Optional[Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]] = None,
        return_metadata: bool = True,
        use_metadata_csv: bool = True,
        check_files: bool = True,
    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.task = task
        self.subset = subset
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.return_metadata = return_metadata
        self.use_metadata_csv = use_metadata_csv
        self.check_files = check_files

        self.task_dir = self.root / self.task / self.subset
        self.input_dir = self.task_dir / "input"
        self.target_dir = self.task_dir / "target"
        self.metadata_path = self.task_dir / "metadata.csv"

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        if not self.target_dir.exists():
            raise FileNotFoundError(f"Target directory not found: {self.target_dir}")

        self.samples = self._build_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples found in task={self.task}, subset={self.subset}"
            )

    def _build_samples(self) -> list[dict]:
        if self.use_metadata_csv and self.metadata_path.exists():
            return self._build_samples_from_csv()
        return self._build_samples_from_dirs()

    def _build_samples_from_csv(self) -> list[dict]:
        samples: list[dict] = []

        with open(self.metadata_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required_cols = {"patch_id", "input_path", "target_path"}
            if not required_cols.issubset(reader.fieldnames or set()):
                raise ValueError(
                    f"metadata.csv 缺少必要字段，至少需要: {required_cols}，"
                    f"当前字段: {reader.fieldnames}"
                )

            seen = set()
            for row in reader:
                if not row['patch_id'] or row['patch_id'] in seen:
                    raise ValueError(f"Empty or duplicate patch_id in {self.metadata_path}: {row['patch_id']}")
                seen.add(row['patch_id'])
                for key, expected in (("task", self.task), ("subset", self.subset)):
                    if row.get(key, expected) != expected:
                        raise ValueError(f"Wrong {key} in {self.metadata_path}: {row.get(key)}")
                input_path = Path(row["input_path"])
                target_path = Path(row["target_path"])

                if not input_path.is_absolute():
                    input_path = (self.task_dir / input_path).resolve()
                if not target_path.is_absolute():
                    target_path = (self.task_dir / target_path).resolve()

                if self.check_files and (not input_path.exists() or not target_path.exists()):
                    raise FileNotFoundError(f"Missing paired files: {input_path}, {target_path}")

                samples.append(
                    {
                        "patch_id": row.get("patch_id", input_path.stem),
                        "fov_key": row.get("fov_key") or re.sub(r"_r\d+_c\d+$", "", row["patch_id"]),
                        "subject_id": row.get('subject_id') or subject_from_key(row['patch_id']),
                        "input_path": input_path,
                        "target_path": target_path,
                        "source_input": row.get('source_input', ''),
                        "source_target": row.get('source_target', ''),
                    }
                )

        return samples

    def _build_samples_from_dirs(self) -> list[dict]:
        input_files = sorted(self.input_dir.glob("*.npy"))
        target_files = sorted(self.target_dir.glob("*.npy"))

        target_map = {p.stem: p for p in target_files}
        input_ids = {p.stem for p in input_files}
        if input_ids != set(target_map):
            raise ValueError(f"Unpaired files in {self.task_dir}: {sorted(input_ids ^ set(target_map))[:5]}")
        samples: list[dict] = []

        for input_path in input_files:
            patch_id = input_path.stem
            target_path = target_map.get(patch_id)

            if target_path is None:
                continue

            samples.append(
                {
                    "patch_id": patch_id,
                    "fov_key": re.sub(r"_r\d+_c\d+$", "", patch_id),
                    "input_path": input_path,
                    "target_path": target_path,
                }
            )

        return samples

    @staticmethod
    def _load_npy(path: Path) -> np.ndarray:
        arr = np.load(path, allow_pickle=False)

        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Loaded object is not a numpy array: {path}")

        if arr.ndim == 2:
            # H, W -> H, W, 1
            arr = arr[..., None]
        elif arr.ndim == 3:
            pass
        else:
            raise ValueError(f"Unsupported array shape {arr.shape} for file: {path}")

        if not np.issubdtype(arr.dtype, np.number) or not np.isfinite(arr).all():
            raise ValueError(f"Non-numeric or non-finite data: {path}")
        if arr.size == 0 or arr.min() < 0 or arr.max() > 1:
            raise ValueError(f"Expected pre-normalized [0, 1] data: {path}")
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _to_tensor_input(arr: np.ndarray) -> Tensor:
        """
        arr: H x W x C
        return: C x H x W
        """
        if arr.ndim != 3:
            raise ValueError(f"Input array must be 3D, got {arr.shape}")
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    @staticmethod
    def _to_tensor_target(arr: np.ndarray) -> Tensor:
        """
        arr: H x W x 1 或 H x W x C
        默认转成 1 x H x W
        """
        if arr.ndim != 3:
            raise ValueError(f"Target array must be 3D, got {arr.shape}")

        if arr.shape[-1] != 1:
            raise ValueError(f"Expected single-channel target, got {arr.shape}")

        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict | tuple[Tensor, Tensor]:
        sample = self.samples[index]

        input_arr = self._load_npy(sample["input_path"])
        target_arr = self._load_npy(sample["target_path"])

        input_tensor = self._to_tensor_input(input_arr)
        target_tensor = self._to_tensor_target(target_arr)
        if input_tensor.shape[-2:] != target_tensor.shape[-2:]:
            raise ValueError(f"Input/target spatial mismatch for {sample['patch_id']}")

        # 数据已经在预处理阶段归一化到 [0, 1]，这里默认不再重复归一化
        if self.joint_transform is not None:
            input_tensor, target_tensor = self.joint_transform(input_tensor, target_tensor)

        if self.input_transform is not None:
            input_tensor = self.input_transform(input_tensor)

        if self.target_transform is not None:
            target_tensor = self.target_transform(target_tensor)

        if not self.return_metadata:
            return input_tensor, target_tensor

        return {
            "input": input_tensor,            # [C, H, W]
            "target": target_tensor,          # [1, H, W]
            "input_path": str(sample["input_path"]),
            "target_path": str(sample["target_path"]),
            "patch_id": sample["patch_id"],
            "fov_key": sample["fov_key"],
            "subject_id": sample.get('subject_id') or subject_from_key(sample['fov_key']),
            "task": self.task,
            "subset": self.subset,
        }

# #%%
# from torch.utils.data import DataLoader

# root = r"E:\Virtual_strain\dataset_patches"

# train_dataset = VirtualStainingPatchDataset(
#     root=root,
#     task="BFtoIBA1",
#     subset="train",
#     return_metadata=False,
# )

# val_dataset = VirtualStainingPatchDataset(
#     root=root,
#     task="BFtoIBA1",
#     subset="val",
#     return_metadata=False,
# )

# test_dataset = VirtualStainingPatchDataset(
#     root=root,
#     task="BFtoIBA1",
#     subset="test",
#     return_metadata=False,
# )
# #%%
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
# val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

# x, y = next(iter(train_loader))
# print(x.shape)  # 例如: [8, 3, 256, 256]
# print(y.shape)  # 例如: [8, 1, 256, 256]
# #%%
# import random
# import torch

# class RandomHorizontalFlipPair:
#     def __init__(self, p: float = 0.5):
#         self.p = p

#     def __call__(self, x: torch.Tensor, y: torch.Tensor):
#         if random.random() < self.p:
#             x = torch.flip(x, dims=[2])  # W 方向
#             y = torch.flip(y, dims=[2])
#         return x, y

# train_dataset = VirtualStainingPatchDataset(
#     root=r"E:\Virtual_strain\dataset_patches",
#     task="BFtoLYVE1",
#     subset="train",
#     joint_transform=RandomHorizontalFlipPair(p=0.5),
#     return_metadata=False,
# )
