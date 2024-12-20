from typing import Any, Iterator

import cv2
import torch
import torch.nn.functional as functional
import solt
import pandas as pd
import numpy as np
import os
import math
import PIL
from pathlib import Path
from natsort import natsorted
from torch.utils.data import Dataset, Sampler
from typing import Optional
# import matplotlib.pyplot as plt

import mlpipeline.utils.common as common
from mlpipeline.losses.boundary_loss import dist_map_transform


def two_branches_dataset(f):
    def wrapper(*args):
        config = args[0].config
        sample = f(*args)

        two_branches = False if (
            config.two_branches is None
        ) else config.two_branches
        two_branches_num_patches = 1 if (
            config.two_branches_num_patches is None
        ) else config.two_branches_num_patches
        if (not two_branches) or (two_branches_num_patches == 1):
            return sample

        inputs = sample["input"]
        assert math.pow(int(math.sqrt(two_branches_num_patches)),
                        2) == two_branches_num_patches
        patch_size = (
            inputs.shape[-2] // int(math.sqrt(two_branches_num_patches)),
            inputs.shape[-1] // int(math.sqrt(two_branches_num_patches)),
        )
        assert inputs.shape[-2] % patch_size[0] == 0
        assert inputs.shape[-1] % patch_size[1] == 0

        original_input = inputs.clone()
        original_input = functional.interpolate(
            original_input.unsqueeze(dim=0),
            patch_size,
            mode="bilinear",
            align_corners=True,
        )

        inputs = torch.split(
            inputs, split_size_or_sections=patch_size[1], dim=2)
        inputs = torch.stack(inputs, dim=0)
        inputs = torch.split(
            inputs, split_size_or_sections=patch_size[0], dim=2)
        inputs = torch.cat(inputs + (original_input,), dim=0)
        sample["input"] = inputs

        return sample

    return wrapper


class DataFrameDataset(Dataset):
    def __init__(
        self, metadata: pd.DataFrame,
        data_key: str = 'data',
        target_key: Optional[str] = 'target',
        **kwargs,
    ):
        self.metadata = metadata
        self.data_key = data_key
        self.target_key = target_key

    def read_data(self, entry):
        return getattr(entry, self.data_key)

    def __getitem__(self, idx):
        entry = self.metadata.iloc[idx]
        # Getting the data from the dataframe
        res = self.read_data(entry)
        res['idx'] = idx
        return res

    def __len__(self):
        return self.metadata.shape[0]


class DataFrameImageDataset(DataFrameDataset):
    def __init__(
        self, metadata: pd.DataFrame, transforms: solt.Stream,
        data_key: str = 'data',
        target_key: Optional[str] = 'target', mean=None, std=None,
        **kwargs,
    ):
        super(DataFrameImageDataset, self).__init__(
            metadata, data_key, target_key)
        self.transforms = transforms
        self.mean = mean
        self.std = std

    def read_data(self, entry):
        img = getattr(entry, self.data_key)
        return self.transforms(img, mean=self.mean, std=self.std)['image']


def check_missing_data(x):
    return (isinstance(x, str) and "missing" in x) or not x or x != x


class FIVESDataset(DataFrameDataset):
    def __init__(
        self, root: str, metadata: pd.DataFrame,
        transforms: solt.Stream, patch_transforms: solt.Stream,
        image_transforms: solt.Stream,
        config: Any,
        data_key: str = 'data',
        target_key: Optional[str] = 'target', mean=None, std=None,
        **kwargs,
    ):
        super().__init__(metadata, data_key, target_key)
        self.transform_lib = 'solt'
        self.transforms = transforms
        self.patch_transforms = patch_transforms
        self.image_transforms = image_transforms
        self.stats = {'mean': mean, 'std': std}
        self.root = root
        self.config = config
        self.disttransform = dist_map_transform([1, 1], 2)

        self.uncertainty_postfix = "" if config.uncertainty_postfix is None else config.uncertainty_postfix

        patch_metadata = metadata.loc[metadata['input'].str.contains(
            "_patches/")]
        self.length_patches = patch_metadata.shape[0]

    def apply_transform(self, img_input, img_gt, img_skeleton=None, uncertainty_labels=None, stage_name=None):
        if self.transform_lib == 'solt':
            if len(img_input.shape) == 2:
                img_input = np.expand_dims(img_input, -1)
            if len(img_gt.shape) == 2:
                img_gt = np.expand_dims(img_gt, -1)
            if img_skeleton is not None:
                if len(img_skeleton.shape) == 2:
                    img_skeleton = np.expand_dims(img_skeleton, -1)
                img_gt = np.concatenate([img_gt, img_skeleton], axis=-1)

            # Preprocessing
            # max(img_gt) is still 255
            if (self.image_transforms is not None):
                tensors = self.image_transforms(
                    {'image': img_input, 'mask': img_gt}, return_torch=True, normalize=False)
                img_input = tensors['image'].permute(1, 2, 0).numpy()
                img_input = (img_input * 255).astype(np.uint8)
                img_gt = tensors['mask']
                img_gt = img_gt.permute(
                    1, 2, 0) if img_gt.ndim == 3 else img_gt.squeeze(dim=0)
                img_gt = img_gt.numpy()

            # Pack GT masks
            if uncertainty_labels is not None:
                if uncertainty_labels.ndim == 2:
                    uncertainty_labels = np.expand_dims(
                        uncertainty_labels, axis=-1)
                if uncertainty_labels.shape[0] < uncertainty_labels.shape[2]:
                    uncertainty_labels = uncertainty_labels.transpose(1, 2, 0)
                img_gt = np.concatenate([img_gt, uncertainty_labels], axis=-1)

            # Run augmentation
            if (stage_name is not None) and (self.patch_transforms is not None) and ("patches" in stage_name):
                trf = self.patch_transforms(
                    {'image': img_input, 'mask': img_gt}, return_torch=True, normalize=True, **self.stats)
            else:
                trf = self.transforms(
                    {'image': img_input, 'mask': img_gt}, return_torch=True, normalize=True, **self.stats)

            # trf_input has shape [N, H, W]
            # trf_gt has shape [1, H, W, N]
            trf_input = trf['image']
            trf_gt = trf['mask']
            if len(trf_gt.shape) == 3:
                trf_gt = torch.unsqueeze(trf_gt, dim=-1)

            # Normalize
            for i in range(0, trf_gt.shape[-1]):
                max_value = trf_gt[..., i].max()

                if max_value > 200:
                    trf_gt[..., i] = np.clip(
                        trf_gt[..., i] / 255.0, a_min=0.0, a_max=1.0)
                elif max_value > 1.5:
                    trf_gt[..., i] = np.clip(
                        trf_gt[..., i] - 1.0, a_min=-1.0, a_max=1.0)
        else:
            raise ValueError(
                f'Not support transform library {self.transform_lib}')

        trf_gt = trf_gt.squeeze(dim=0)
        # Unpack GT masks
        trf_skeleton = trf_uncertainty = None
        num_classes = self.config.num_classes if uncertainty_labels is None else 1
        index = num_classes
        if img_skeleton is not None:
            trf_skeleton = trf_gt[:, :, index]
            index += 1
        if (uncertainty_labels is not None) or (trf_gt.shape[-1] > num_classes and img_skeleton is None):
            end_index = trf_gt.shape[-1] if uncertainty_labels is None else (
                index + uncertainty_labels.shape[2])
            trf_uncertainty = trf_gt[:, :, index:end_index]

        trf_gt = trf_gt[:, :, :num_classes]

        if trf_input.ndim == 2:
            trf_input = torch.unsqueeze(trf_input, dim=0)
        if trf_gt.ndim == 2:
            trf_gt = torch.unsqueeze(trf_gt, dim=0)
        else:
            trf_gt = trf_gt.permute(2, 0, 1)

        if (trf_skeleton is not None) and len(trf_skeleton.shape) == 2:
            trf_skeleton = torch.unsqueeze(trf_skeleton, dim=0)
        if (trf_uncertainty is not None) and len(trf_uncertainty.shape) == 2:
            trf_uncertainty = torch.unsqueeze(trf_uncertainty, dim=0)
        elif trf_uncertainty is not None:
            trf_uncertainty = trf_uncertainty.permute(2, 0, 1)

        trf_gt = trf_gt.long()
        return trf_input, trf_gt, trf_skeleton, trf_uncertainty

    @two_branches_dataset
    def read_data(self, entry):
        sample = {}

        filename_input = entry['input']
        fullname_input = os.path.join(self.root['fives'], filename_input)

        filename_gt = entry['gt']
        fullname_gt = os.path.join(self.root['fives'], filename_gt)
        img_gt = common.read_image(fullname_gt, gray=True)

        # Read input
        img_input = common.read_image(
            fullname_input, gray=(self.config.num_channels == 1))

        if img_input is None:
            raise ValueError(f'Not found image file {fullname_input}.')
        if img_gt is None:
            raise ValueError(f'Not found image file {fullname_gt}.')

        img_skeleton = uncertainty_labels = None
        # Get skeleton for SkelCon
        if self.config.use_skeleton:
            filename_skeleton = filename_input.replace("Original", "Skeleton")
            fullname_skeleton = os.path.join(
                self.root['fives'], filename_skeleton)
            img_skeleton = common.read_image(fullname_skeleton, gray=True)
        # Get uncertainty labels
        if (self.config.use_uncertainty is not None) and (len(self.config.use_uncertainty) > 0):
            use_uncertainty = self.config.use_uncertainty.split(",")
            uncertainty_labels = []
            postfix = "_" + \
                self.uncertainty_postfix if len(
                    self.uncertainty_postfix) > 0 else ""

            for label_name in use_uncertainty:
                filename_label = filename_gt.replace(
                    "GroundTruth", f"GroundTruth_{label_name}{postfix}")
                filename_label = filename_label.replace(".png", ".npy")
                fullname_label = os.path.join(
                    self.root['fives'], filename_label)
                uncertainty_label = np.load(fullname_label)

                if uncertainty_label.ndim == 2:
                    uncertainty_label = np.expand_dims(
                        uncertainty_label, axis=-1)
                for i in range(0, uncertainty_label.shape[2]):
                    if uncertainty_label[..., i].min() < -0.9:
                        uncertainty_label[...,
                                          i] = uncertainty_label[..., i] + 1.0

                # min_pos = np.min(uncertainty_label[uncertainty_label > 1])
                # max_neg = np.max(uncertainty_label[uncertainty_label < 1])
                # print(filename_input, min_pos, max_neg)
                uncertainty_labels.append(uncertainty_label)

            uncertainty_labels = np.concatenate(uncertainty_labels, axis=-1)

        stage_name = Path(filename_input).parent.parent.stem
        trf_img_input, trf_img_gt, trf_img_skeleton, trf_uncertainty = self.apply_transform(
            img_input, img_gt,
            img_skeleton, uncertainty_labels,
            stage_name)

        if trf_uncertainty is not None:
            trf_img_gt = torch.cat([trf_img_gt, trf_uncertainty], dim=0)

        sample = {}
        sample['input'] = trf_img_input
        sample['gt'] = trf_img_gt
        sample['paths'] = [filename_input, filename_gt]

        if trf_img_skeleton is not None:
            sample['skeleton'] = trf_img_skeleton

        if trf_uncertainty is None:
            sample['dist_map'] = self.disttransform(trf_img_gt)

        return sample


class DriveDataset(DataFrameDataset):
    def __init__(
        self, root: str, metadata: pd.DataFrame,
        transforms: solt.Stream, patch_transforms: solt.Stream,
        config: Any,
        data_key: str = 'data',
        target_key: Optional[str] = 'target', mean=None, std=None,
        **kwargs,
    ):
        super().__init__(metadata, data_key, target_key)
        self.transform_lib = 'solt'
        self.transforms = transforms
        self.patch_transforms = patch_transforms
        self.stats = {'mean': mean, 'std': std}
        self.root = root
        self.config = config

    def apply_transform(self, img_input, img_gt, stage_name=None):
        if self.transform_lib == 'solt':
            if len(img_input.shape) == 2:
                img_input = np.expand_dims(img_input, -1)
            if len(img_gt.shape) == 2:
                img_gt = np.expand_dims(img_gt, -1)

            if (stage_name is not None) and (self.patch_transforms is not None) and ("patches" in stage_name):
                trf = self.patch_transforms(
                    {'image': img_input, 'mask': img_gt}, return_torch=True, normalize=True, **self.stats)
            else:
                trf = self.transforms(
                    {'image': img_input, 'mask': img_gt}, return_torch=True, normalize=True, **self.stats)

            trf_input = trf['image']
            trf_gt = trf['mask'] / 255.0
        else:
            raise ValueError(
                f'Not support transform library {self.transform_lib}')

        if len(trf_input.shape) == 2:
            trf_input = torch.unsqueeze(trf_input, dim=1)
        if len(trf_gt.shape) == 2:
            trf_gt = torch.unsqueeze(trf_gt, dim=1)
        return trf_input, trf_gt

    @two_branches_dataset
    def read_data(self, entry):
        sample = {}

        filename_input = entry['input']
        filename_gt = entry["gt"]
        fullname_input = os.path.join(self.root['drive'], filename_input)
        fullname_gt = os.path.join(self.root['drive'], filename_gt)
        img_input = common.read_image(
            fullname_input, gray=(self.config.num_channels == 1))
        img_gt = common.read_image(fullname_gt, gray=True)

        if img_input is None:
            raise ValueError(f'Not found image file {fullname_input}.')

        if img_gt is None:
            raise ValueError(f'Not found image file {fullname_gt}.')

        stage_name = Path(filename_input).parent.parent.stem
        trf_img_input, trf_img_gt = self.apply_transform(
            img_input, img_gt, stage_name)

        sample = {}
        sample['input'] = trf_img_input
        sample['gt'] = trf_img_gt
        sample['paths'] = [filename_input, filename_gt]

        return sample


class MixedTestDataset(Dataset):
    def __init__(self, metadata, config):
        super(MixedTestDataset, self).__init__()
        self.metadata = metadata
        self.config = config

        self.root_dir = config.image_dir.base
        self.transforms = solt.utils.from_yaml(config.augs.test if (
            config.augs.test is not None) else {"stream": {"transforms": []}})
        self.stats = {"mean": tuple(config.mean), "std": tuple(config.std)}

        self.items = self._load_datasets()

    def _load_datasets(self):
        items = []

        for dataset_name, metadata in self.metadata.items():
            input_dir = Path(self.root_dir) / metadata.input_dir
            input_paths = natsorted(
                input_dir.glob(f"*.{metadata['input_ext']}"))
            gt_dir = Path(self.root_dir) / metadata.gt_dir
            gt_paths = natsorted(gt_dir.glob(f"*.{metadata['gt_ext']}"))

            pairs = [
                {
                    "input": str(input_path),
                    "gt": str(gt_path),
                    "dataset": dataset_name,
                }
                for (input_path, gt_path) in zip(input_paths, gt_paths)
            ]
            items += pairs

        return items

    def __len__(self):
        return len(self.items)

    def apply_transform(self, img_input, img_gt):
        if len(img_input.shape) == 2:
            img_input = np.expand_dims(img_input, -1)
        if len(img_gt.shape) == 2:
            img_gt = np.expand_dims(img_gt, -1)

        trf = self.transforms({"image": img_input, "mask": img_gt},
                              return_torch=True, normalize=True, **self.stats)
        trf_input = trf["image"]
        trf_gt = trf["mask"] / 255.0
        return trf_input, trf_gt

    @two_branches_dataset
    def __getitem__(self, index):
        item = self.items[index]
        input_path, gt_path = item["input"], item["gt"]
        dataset_name = item["dataset"]

        img_input = common.read_image(
            input_path, gray=(self.config.num_channels == 1))
        img_gt = common.read_image(gt_path, gray=True)
        trf_img_input, trf_img_gt = self.apply_transform(img_input, img_gt)

        sample = {}
        sample["input"] = trf_img_input
        sample["gt"] = trf_img_gt
        sample["dataset"] = dataset_name
        sample["paths"] = [input_path, gt_path]
        return sample


class DataFrameMultilabelImageDataset(DataFrameImageDataset):
    def __getitem__(self, idx):
        res = DataFrameImageDataset.__getitem__(self, idx)
        res["target"] = torch.from_numpy(res["target"]).float()
        return res


class ImageFolderDataset(DataFrameImageDataset):
    def __init__(self, metadata: pd.DataFrame, transforms: solt.Stream,
                 data_key: str = 'data',
                 target_key: Optional[str] = 'target', mean=None, std=None):
        super(ImageFolderDataset, self).__init__(
            metadata, transforms, data_key, target_key, mean, std)

    def read_data(self, entry) -> dict:
        img = cv2.imread(str(getattr(entry, self.data_key)))
        return self.transforms(img, mean=self.mean, std=self.std)['image']


class PatchWholeSampler(Sampler[int]):
    r"""
    Args:
        data_source (Dataset): dataset to sample from
        generator (Generator): Generator used in sampling.
    """

    def __init__(
        self, data_source: FIVESDataset,
        generator=None,
    ) -> None:
        self.data_source = data_source
        self.generator = generator
        # Get number of whole images from the original dataset
        self.source_num_samples = data_source.config.training_samples
        self.source_num_samples = self.source_num_samples if self.source_num_samples > -1 else 480
        self.fixed_ratio = 0.5

        # Get a random quarter of ready-made patches
        self.num_patch_samples = int(
            (data_source.length_patches // self.source_num_samples) * self.fixed_ratio) * self.source_num_samples
        self._num_samples = self.num_patch_samples + \
            (len(data_source) - data_source.length_patches)

        self.input = torch.tensor(
            [1] * 49 + [3] * 25 + [2] * 25 + [2] * 30 + [0] * 25, dtype=torch.float32)
        self.input = torch.cat([self.input] * self.source_num_samples, dim=0)

        self.whole_indices = torch.nonzero(
            self.input <= 0, as_tuple=False)[:, 0]
        self.all_indices = torch.arange(0, len(self.data_source))

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # self.patch_indices = torch.nonzero(self.input > 0, as_tuple=False)[:, 0]
        # print(self.patch_indices.shape, self.whole_indices.shape, self.all_indices.shape)
        sampled_patch_indices = self.all_indices[torch.multinomial(
            input=self.input,
            num_samples=self.num_patch_samples,
            replacement=False,
            generator=generator,
        )]
        sampled_patch_indices = torch.sort(sampled_patch_indices)[0]

        indices = torch.cat(
            [sampled_patch_indices, self.whole_indices], dim=-1)
        # print(indices.shape, self.num_samples, indices.max())
        indices = indices[torch.randperm(
            self.num_samples,
            generator=generator,
        )].long()
        # print(len(indices), indices[:5], indices.max(), len(sampled_patch_indices), sampled_patch_indices[-5:], sampled_patch_indices.max())

        yield from indices.tolist()

    def __len__(self) -> int:
        return self.num_samples
