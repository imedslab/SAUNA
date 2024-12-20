import os

import gc
import logging
import solt
import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import socket
import pandas as pd

from datetime import datetime
from omegaconf import OmegaConf
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from logging import config as init_logging_config
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader

from mlpipeline.utils.common import init_obj_cls, init_obj
from mlpipeline.data.data_provider import DataProvider
from mlpipeline.train.scheduler import LRScheduler
from mlpipeline.train.checkpointer import Checkpointer
from mlpipeline.utils.eval import accuracy


class MLPipeline:

    pipeline_name = 'mlpipeline'

    def __init__(self, cfg, local_rank, global_rank):
        self.cfg = cfg
        self.val_loader = None
        self.train_loader = None
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.models = None
        self.epoch = 0
        self.lr_scheduler = None
        self.model, self.optimizer, self.criterion = None, None, None
        self.original_cwd = Path(cfg.original_cwd)
        # Changing cwd to the autogenerated snapshot dir as defined in the config
        os.chdir(cfg.snapshot_dir)

        self.cache_dir = None
        self.log_writer = None
        self.logger = None
        self.checkpointer = None

        self.distributed = self.cfg.train.distributed

    def init_splits(self):
        if self.cfg.data.data_dir is None:
            self.cfg.data.data_dir = str(self.original_cwd / 'datasets')

        data_provider = DataProvider(
            self.cfg,
            self.logger,
            self.global_rank,
            self.distributed)

        return data_provider.init_splits()

    def post_process_dataframes(self, train_df, test_df):
        return train_df, test_df

    def init_loaders(self):
        train_df, val_df = self.init_splits()
        if self.global_rank == 0:
            self.logger.info(
                f'Before post-processing, there are {len(train_df.index)}/{len(val_df.index)} samples for training/validation')
            train_df, val_df = self.post_process_dataframes(train_df, val_df)
            self.logger.info(
                f'After post-processing, there are {len(train_df.index)}/{len(val_df.index)} samples for training/validation')
            train_df.to_pickle('train.pkl')
            val_df.to_pickle('val.pkl')

        if 0 < self.cfg.data.subsample < 1:
            if self.global_rank == 0:
                train_df, unused_data = train_test_split(
                    train_df,
                    train_size=self.subsample,
                    shuffle=True,
                    random_state=self.seed,
                    stratify=train_df.target,
                )
                train_df.to_pickle('train.pkl')
                unused_data.to_pickle('unused_train.pkl')
                val_df.to_pickle('val.pkl')

        self.barrier()

        train_df = pd.read_pickle('train.pkl')
        val_df = pd.read_pickle('val.pkl')

        train_transforms = solt.utils.from_yaml(self.cfg.data.augs.train)
        if self.cfg.data.augs.train_patches is not None:
            train_patches_transforms = solt.utils.from_yaml(
                self.cfg.data.augs.train_patches)
        else:
            train_patches_transforms = None
        if self.cfg.data.augs.image is not None:
            image_transforms = solt.utils.from_yaml(self.cfg.data.augs.image)
        else:
            image_transforms = None
        if self.cfg.data.augs.val is not None:
            val_transforms = solt.utils.from_yaml(self.cfg.data.augs.val)
        else:
            val_transforms = solt.Stream()

        mean = None
        if self.cfg.data.mean is not None:
            assert len(
                self.cfg.data.mean) == self.cfg.data.num_channels, f"Mean needs to be an iterable of length {self.cfg.data.num_channels}"
            mean = tuple(self.cfg.data.mean)

        std = None
        if self.cfg.data.std is not None:
            assert len(
                self.cfg.data.std) == self.cfg.data.num_channels, f"Std needs to be an iterable of length {self.cfg.data.num_channels}"
            std = tuple(self.cfg.data.std)

        dataset_cls = init_obj_cls(self.cfg.data.dataset_cls)
        train_ds = dataset_cls(
            root=self.cfg.data.image_dir, metadata=train_df, transforms=train_transforms, mean=mean, std=std,
            patch_transforms=train_patches_transforms,
            image_transforms=image_transforms,
            image_mode=self.cfg.data.image_mode,
            config=self.cfg.data,
            data_key=self.cfg.data.data_key if 'data_key' in self.cfg.data else None,
            target_key=self.cfg.data.target_key if 'target_key' in self.cfg.data else None)
        val_ds = dataset_cls(
            root=self.cfg.data.image_dir, metadata=val_df, transforms=val_transforms, mean=mean, std=std,
            patch_transforms=None,
            image_transforms=image_transforms,
            image_mode=self.cfg.data.image_mode,
            config=self.cfg.data,
            data_key=self.cfg.data.data_key if 'data_key' in self.cfg.data else None,
            target_key=self.cfg.data.target_key if 'target_key' in self.cfg.data else None)

        train_sampler, val_sampler = self.init_samplers(train_ds, val_ds)

        self.train_loader = DataLoader(
            train_ds, batch_size=self.cfg.data.batch_size, shuffle=(
                train_sampler is None),
            num_workers=self.cfg.data.num_workers,
            pin_memory=False, sampler=train_sampler)

        if self.cfg.data.valid_batch_size is None:
            valid_batch_size = self.cfg.data.batch_size
        else:
            valid_batch_size = self.cfg.data.valid_batch_size
        self.val_loader = DataLoader(
            val_ds, batch_size=valid_batch_size, shuffle=False,
            num_workers=self.cfg.data.num_workers, pin_memory=True, sampler=val_sampler, drop_last=False)

    def init_samplers(self, train_ds, val_ds):
        train_sampler, val_sampler = None, None
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_ds, rank=self.global_rank)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_ds, rank=self.global_rank)

        return train_sampler, val_sampler

    def init_tensorboard(self):
        logdir = self.cache_dir / 'tb_logs' / 'run'
        logdir.mkdir(parents=True, exist_ok=True)
        self.log_writer = SummaryWriter(logdir)

    def cleanup(self):
        del self.model, self.optimizer, self.criterion
        del self.checkpointer
        gc.collect()
        torch.cuda.empty_cache()

    def display_rank(self):
        return f'[GPU {self.global_rank}/{self.cfg.world_size-1}]'

    def create_model(self):
        self.model = init_obj(self.cfg.model.cls, self.cfg.model.params)

    def visualize_model(self):
        if not self.distributed or self.global_rank == 0:
            print(self.model)

    def freeze_modules(self):
        pass

    def init_model(self):
        self.create_model()

        if self.distributed:
            load_mode = 'encoder'
            if "pretrained_encoder" in self.cfg.model and self.cfg.model.pretrained_encoder is not None and os.path.isfile(self.cfg.model.pretrained_encoder):
                model_dumpster = self.cfg.model.pretrained_encoder
                self.logger.info(
                    f'{self.display_rank()}: Loading pretrained ENCODER {model_dumpster}')
                raise ValueError()
            elif "pretrained_model" in self.cfg.model and self.cfg.model.pretrained_model is not None and os.path.isfile(self.cfg.model.pretrained_model):
                load_mode = 'all'
                model_dumpster = self.cfg.model.pretrained_model
                self.logger.info(
                    f'{self.display_rank()}: Loading pretrained MODEL {model_dumpster}')
                raise ValueError()
            else:
                if 'TMPDIR' in os.environ and os.path.isdir(os.environ['TMPDIR']):
                    tmp_root = Path(os.environ['TMPDIR'])
                else:
                    tmp_root = Path('/tmp/')

                model_dumpster = tmp_root / \
                    f'tmp_{self.cfg.experiment_setting}_{self.cfg.seed}.pth'

                if self.global_rank == 0:
                    self.logger.info(f'Initialized {self.cfg.model.cls}')
                    state = self.model.state_dict()
                    torch.save(state, model_dumpster)
                self.barrier()

            if load_mode == "encoder":
                self.model.encoder.load_state_dict(torch.load(model_dumpster))
            else:
                self.model.load_state_dict(torch.load(model_dumpster))
                self.logger.info(
                    f'{self.display_rank()} Reset linear layer weights')

            if self.global_rank == 0:
                model_dumpster.unlink()
                self.logger.info(
                    'All processes will start from the same random initialization')
            self.barrier()
        else:
            if "pretrained_encoder" in self.cfg.model and isinstance(self.cfg.model.pretrained_encoder, str) and os.path.isfile(self.cfg.model.pretrained_encoder):
                model_dumpster = self.cfg.model.pretrained_encoder
                self.logger.info(
                    f'{self.display_rank()} Loading pretrained ENCODER {model_dumpster}')
                self.model.encoder.load_state_dict(
                    torch.load(model_dumpster), strict=False)
            elif "pretrained_model" in self.cfg.model and isinstance(self.cfg.model.pretrained_model, str) and os.path.isfile(self.cfg.model.pretrained_model):
                model_dumpster = self.cfg.model.pretrained_model
                self.logger.info(
                    f'{self.display_rank()} Loading pretrained MODEL {model_dumpster}')
                self.model.load_state_dict(
                    torch.load(model_dumpster), strict=True)
                self.logger.info(
                    f'{self.display_rank()} Reset linear layer weights')
            # else:
            #     raise ValueError(f'Not found pretrained model {self.cfg.model.pretrained_encoder} or {self.cfg.model.pretrained_model}.')

        self.freeze_modules()

        self.model.cuda(self.local_rank)
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[
                                                                   self.local_rank, ], find_unused_parameters=self.cfg.model.find_unused_parameters)
            self.barrier()

    def init_logging(self):
        if self.global_rank == 0:
            now = datetime.now()
            dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")
            # Path(f'{self.pipeline_name}_cache_{dt_string}')
            self.cache_dir = Path('saved_model')
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            logging_conf = self.cfg.logging
            log_path = str(self.original_cwd /
                           self.cfg.snapshot_dir / 'runs.log')
            logging_conf.handlers.file.filename = log_path
            logging_conf = OmegaConf.to_container(logging_conf, resolve=True)
            init_logging_config.dictConfig(logging_conf)
            self.logger = logging.getLogger(__name__)

            self.logger.info(f'Host: {socket.gethostname()}')
            self.logger.info(f'Distributed: {self.cfg.train.distributed}')
            self.logger.info(f'Writing log to {log_path}')
            if 'SLURM_JOBID' in os.environ:
                self.logger.info(
                    f'Running slurm job: {os.environ["SLURM_JOBID"]}')
            self.init_tensorboard()

        if self.distributed:
            self.barrier()

    def create_criterion(self):
        self.criterion = init_obj(
            self.cfg.criterion.cls, self.cfg.criterion.params)

    def init_run(self, epoch):
        self.cleanup()
        self.init_logging()
        self.create_criterion()
        self.init_model()

        # Optimizer
        opt_params = dict(self.cfg.optimizer.params)
        opt_params['params'] = self.model.parameters()
        self.optimizer = init_obj(self.cfg.optimizer.cls, opt_params)
        self.lr_scheduler = LRScheduler(self.cfg, self.optimizer, epoch)

        self.init_loaders()
        self.checkpointer = Checkpointer(self)
        self.barrier()

    def train(self):
        self.init_run(0)

        for self.epoch in range(self.cfg.train.num_epochs):
            if self.distributed:
                self.train_loader.sampler.set_epoch(self.epoch)
            self.lr_scheduler.step(self.epoch)

            self.model.train()
            train_loss = self.train_epoch()

            self.model.eval()
            val_loss, val_accuracy = self.val_epoch()

            if self.global_rank == 0:
                self.checkpointer.save_state(val_accuracy)
                self.log_writer.add_scalars(
                    'Loss', {'train': train_loss, 'val': val_loss},
                    global_step=self.epoch)
                self.log_writer.add_scalar(
                    f'Metrics/accuracy', val_accuracy,
                    global_step=self.epoch)

                # Logging the metrics
                if self.global_rank == 0:
                    log_out = f"[Epoch {self.epoch}] lr: {self.lr_scheduler.lr:.4f}"
                    log_out += f"--train loss: {train_loss:.4f}"
                    log_out += f"--val loss: {val_loss:.4f}"
                    log_out += f"--val acc: {val_accuracy:.4f}"
                    self.logger.info(log_out)

            self.barrier()
        return

    def train_epoch(self):
        if self.global_rank == 0:
            pbar = tqdm(total=len(self.train_loader))
        else:
            pbar = None

        running_loss = torch.tensor(
            0., requires_grad=False).cuda(self.local_rank)

        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            images = batch['data'].cuda(self.local_rank, non_blocking=True)
            target = batch['target'].cuda(self.local_rank, non_blocking=True)

            output = self.model(images)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            cur_loss = running_loss.item() / (i + 1)

            if self.global_rank == 0:
                desc = f'[{self.epoch}] Train {loss.item():.4f} / {cur_loss:.4f}'
                pbar.set_description(desc)
                pbar.update()

        if self.global_rank == 0:
            pbar.close()
        if self.distributed:
            dist.all_reduce(running_loss)

        running_loss = running_loss / self.cfg.n_gpus / len(self.train_loader)
        return running_loss.item()

    def val_epoch(self):
        if self.global_rank == 0:
            pbar = tqdm(total=len(self.val_loader))
        else:
            pbar = None

        running_loss = torch.tensor(
            0., requires_grad=False).cuda(self.local_rank)
        running_accuracy = torch.tensor(
            0., requires_grad=False).cuda(self.local_rank)

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                images = batch['data'].cuda(self.local_rank, non_blocking=True)
                target = batch['target'].cuda(
                    self.local_rank, non_blocking=True)

                output = self.model(images)
                loss = self.criterion(output, target)

                running_loss.add_(loss)
                acc = accuracy(
                    output, target, multilabel=self.cfg.data.setting == "multilabel")
                running_accuracy.add_(acc)

                if self.global_rank == 0:
                    desc = f'[{self.epoch}] Val'
                    pbar.set_description(desc)
                    pbar.update()
            if self.distributed:
                dist.all_reduce(running_loss)
                dist.all_reduce(running_accuracy)

            loss_total = running_loss / self.cfg.n_gpus / len(self.val_loader)
            accuracy_total = running_accuracy / \
                self.cfg.n_gpus / len(self.val_loader)

        if self.global_rank == 0:
            pbar.close()

        self.barrier()
        return loss_total.item(), accuracy_total.item()

    def barrier(self):
        if self.distributed:
            dist.barrier()
        return

    def infer(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError
