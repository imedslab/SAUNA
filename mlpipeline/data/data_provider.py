import torch.distributed as dist
import pandas as pd
import cv2
import torch
import os
import numpy as np
import pickle
from pathlib import Path
from natsort import natsorted
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.datasets.utils import download_url

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def multi_class_split_dataset(method):
    def wrapper(self, data_folder, train=True):
        metadata, n_cls = method(self, data_folder, train)
        if train:
            self.log.info(f'Making a train-val split ({self.val_amount})')
            train_df, val_df = train_test_split(
                metadata,
                test_size=self.val_amount,
                shuffle=True,
                random_state=self.seed,
                stratify=metadata.target,
            )
            return train_df, val_df, None, n_cls
        return None, None, metadata, n_cls
    return wrapper


def wrap_channels(imgs):
    if len(imgs.shape) == 3:
        return np.stack((imgs[:, :, :, None], imgs[:, :, :, None], imgs[:, :, :, None]), axis=3).squeeze()
    return imgs


def make_image_target_df(imgs, labels):
    list_rows = [
        {"data": imgs[i, :, :, :], "target": labels[i]}
        for i in range(len(labels))
    ]
    return pd.DataFrame(list_rows)


class DataProvider(object):
    allowed_datasets = ["fives", "drive"]
    in_memory_datasets = ["fives", "drive"]

    def __init__(self, cfg, logger, rank=0, distributed=False):
        if cfg.data.dataset not in self.allowed_datasets:
            raise ValueError(f"Unsupported dataset {cfg.data.dataset}")

        self.cfg = cfg
        self.val_amount = cfg.data.val_amount
        self.dataset = cfg.data.dataset
        self.seed = cfg.seed
        self.data_folder = cfg.data.data_dir
        os.makedirs(self.data_folder, exist_ok=True)
        self.metadata = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.transforms = None
        self.rank = rank
        self.log = logger
        self.distributed = distributed

    def _map_fives_test_df_row(self, row):
        if not self.cfg.train.use_patches:
            return row

        input_path = row["input"]
        gt_path = row["gt"]
        input_name = Path(input_path).stem
        input_dirname = Path(input_path).parent.stem
        gt_dirname = Path(gt_path).parent.stem

        input_patch_paths = natsorted(Path(
            f"{self.cfg.data.image_dir.fives}/valid_patches/{input_dirname}").glob(f"{input_name}_*.png"))
        gt_patch_paths = natsorted(Path(
            f"{self.cfg.data.image_dir.fives}/valid_patches/{gt_dirname}").glob(f"{input_name}_*.png"))

        row["input"] = [
            str(p.relative_to(self.cfg.data.image_dir.fives))
            for p in input_patch_paths]
        row["gt"] = [
            str(p.relative_to(self.cfg.data.image_dir.fives))
            for p in gt_patch_paths]
        # Length 4
        return row

    def _map_fives_train_df_row(self, row):
        if not self.cfg.train.use_patches:
            return row

        input_path = row["input"]
        gt_path = row["gt"]
        input_name = Path(input_path).stem
        input_dirname = Path(input_path).parent.stem
        gt_dirname = Path(gt_path).parent.stem

        input_patch_paths = natsorted(Path(
            f"{self.cfg.data.image_dir.fives}/train_patches/{input_dirname}").glob(f"{input_name}_*.png"))
        gt_patch_paths = natsorted(Path(
            f"{self.cfg.data.image_dir.fives}/train_patches/{gt_dirname}").glob(f"{input_name}_*.png"))
        # Length 129

        row["input"] = [
            str(p.relative_to(self.cfg.data.image_dir.fives))
            for p in input_patch_paths]
        row["gt"] = [
            str(p.relative_to(self.cfg.data.image_dir.fives))
            for p in gt_patch_paths]

        row["input"] += [input_path] * 25
        row["gt"] += [gt_path] * 25
        # Length 160
        return row

    def init_fives(self, *args, **kwargs):
        fullname = os.path.join(self.data_folder, self.cfg.data.pkl_filename)
        if os.path.isfile(fullname):
            with open(fullname, 'rb') as f:
                data = pickle.load(f)
            train_df, test_df = data[self.cfg.data.fold_index]
        else:
            raise ValueError(f'{fullname} not found.')

        num_train_samples = num_valid_samples = -1
        if self.cfg.data.valid_samples is not None:
            num_valid_samples = self.cfg.data.valid_samples
        if self.cfg.data.training_samples is not None:
            if self.cfg.data.valid_samples is None:
                num_valid_samples = self.cfg.data.training_samples
            num_train_samples = self.cfg.data.training_samples
        train_df = train_df if num_train_samples == - \
            1 else train_df.iloc[:num_train_samples]
        test_df = test_df if num_valid_samples == - \
            1 else test_df.iloc[:num_valid_samples]

        # Transform df
        if self.cfg.train.use_patches:
            train_df = train_df.apply(self._map_fives_train_df_row, axis=1).explode(
                ["input", "gt"]).reset_index(drop=True)

        return train_df, test_df, None, None

    def _map_drive_train_df_row(self, row):
        input_path = row["input"]
        gt_path = row["gt"]

        if self.cfg.train.use_patches:
            input_name = Path(input_path).stem
            gt_name = Path(gt_path).stem
            input_dirname = Path(input_path).parent.stem
            gt_dirname = Path(gt_path).parent.stem

            input_patch_paths = natsorted(Path(
                f"{self.cfg.data.image_dir.drive}/training_patches/{input_dirname}").glob(f"{input_name}_*.tif"))
            gt_patch_paths = natsorted(Path(
                f"{self.cfg.data.image_dir.drive}/training_patches/{gt_dirname}").glob(f"{gt_name}_*.gif"))

            row["input"] = [
                str(p.relative_to(self.cfg.data.image_dir.drive))
                for p in input_patch_paths]
            row["gt"] = [
                str(p.relative_to(self.cfg.data.image_dir.drive))
                for p in gt_patch_paths]

            row["input"] += [input_path] * 2000
            row["gt"] += [gt_path] * 2000

        else:
            row["input"] = [input_path] * 15
            row["gt"] = [gt_path] * 15

        return row

    def init_drive(self, *args, **kwargs):
        fullname = os.path.join(self.data_folder, self.cfg.data.pkl_filename)
        if os.path.isfile(fullname):
            with open(fullname, 'rb') as f:
                data = pickle.load(f)
            train_df, test_df = data[self.cfg.data.fold_index]
        else:
            raise ValueError(f'{fullname} not found.')

        # Transform df
        train_df = train_df.apply(self._map_drive_train_df_row, axis=1).explode(
            ["input", "gt"]).reset_index(drop=True)

        return train_df, test_df, None, None

    def init_splits(self):
        if self.dataset == 'fives':
            if self.rank == 0:
                self.log.info(
                    f'Getting {self.dataset} from {self.data_folder}')
            train_df, val_df, _, _ = getattr(self, f"init_{self.dataset}")()
        elif self.dataset == 'drive':
            if self.rank == 0:
                self.log.info(
                    f'Getting {self.dataset} from {self.data_folder}')
            train_df, val_df, _, _ = getattr(self, f"init_{self.dataset}")()
        else:
            raise ValueError(f'Not support dataset {self.dataset}')

        if self.rank == 0:
            self.log.info(
                f"The split has been loaded from disk by all processes")
        if self.distributed:
            dist.barrier()

        return train_df, val_df
