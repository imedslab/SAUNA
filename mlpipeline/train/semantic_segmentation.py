import os
import re
import time
import json
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.nn.functional as functional
import numpy as np
import cv2

from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data.dataloader import DataLoader

from mlpipeline.train.pipeline import MLPipeline
from mlpipeline.models.semantic_segmentation import SemanticSegmentation as SemanticSegmentationModel
from mlpipeline.metrics.metric_collectors import SemanticSegmentationMetricsCollector
from mlpipeline.data.dataset import MixedTestDataset, PatchWholeSampler
from mlpipeline.samplers.inferer import MySlidingInferer


class SemanticSegmentation(MLPipeline):
    pipeline_name = 'mae'

    def __init__(self, cfg, local_rank, global_rank):
        super().__init__(cfg, local_rank, global_rank)
        if torch.cuda.is_available():
            self.device = torch.device("cuda", local_rank)
        else:
            self.device = torch.device("cpu")
        return

    def init_metrics(self):
        self.metrics_collector = SemanticSegmentationMetricsCollector(
            local_rank=self.local_rank,
            cfg=self.cfg,
        )
        self.key_metric = self.cfg.metrics.get("key_metric", "F1")

    def init_model(self):
        self.create_model()
        # self.freeze_modules()
        self.model.cuda(self.local_rank)
        self.logger.info(f"Device: {str(self.device)}")

        if self.cfg.model.pretrained_model is not None:
            self.model.load_state_dict(torch.load(
                self.cfg.model.pretrained_model,
                map_location=self.device)["model"])
        if self.distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank,],
                find_unused_parameters=self.cfg.model.find_unused_parameters,
            )
            self.barrier()

        self.test_metadata = OmegaConf.load(str(
            Path(self.cfg.data.image_dir.base) / self.cfg.data.test_metadata))
        if self.cfg.model.params.cfg.test_on_patches:
            self.valid_inferer = MySlidingInferer(
                roi_size=self.test_metadata["FIVES"].patch_size,
                input_size=(self.cfg.data.image_size,
                            self.cfg.data.image_size),
                sw_batch_size=1,
                overlap=0.5,
                sw_device=self.device,
                device=self.device,
            )
        return

    def init_samplers(self, train_ds, val_ds):
        if self.cfg.train.use_patches:
            train_sampler = PatchWholeSampler(train_ds)
        else:
            train_sampler = None

        val_sampler = None
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_ds, rank=self.global_rank)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_ds, rank=self.global_rank)

        if train_sampler is not None:
            self.logger.info(
                f"After sampling, there are {len(train_sampler)} samples for training")
        return train_sampler, val_sampler

    def load_continuing_run(self):
        if not self.cfg.train.continue_train:
            return
        if not (Path.cwd() / self.cache_dir).exists():
            return

        weight_paths = sorted(
            list((Path.cwd() / self.cache_dir).glob("*.pth")))
        if len(weight_paths) == 0:
            self.logger.info(f"No checkpoint from previous run found!")
            return

        weight_path = Path(self.cache_dir) / weight_paths[-1]
        self.logger.info(f"{weight_path} {str(os.path.exists(weight_path))}")
        # Get epoch
        epoch = re.findall(f"epoch_(.*?)_", weight_path.stem)[0]
        try:
            epoch = int(epoch)
        except ValueError:
            self.logger.info(
                f"Invalid checkpoint name {weight_path.name} and {epoch}!")
            return
        # Get metrics
        metric = weight_path.stem[(weight_path.stem.rfind("_") + 1):]
        try:
            metric = float(metric)
        except ValueError:
            self.logger.info(
                f"Invalid checkpoint name {weight_path.name} and {metric}!")
            return
        self.logger.info(
            f"Loading model from checkpoint {weight_path} of previous run.")

        # Load model
        checkpoint_data = torch.load(
            str(weight_path),
            map_location=self.device)
        checkpoint_model = checkpoint_data["model"] if "model" in checkpoint_data else checkpoint_data
        if self.distributed:
            self.model.module.load_state_dict(checkpoint_model)
        else:
            self.model.load_state_dict(checkpoint_model)

        if "optimizer" in checkpoint_data:
            self.optimizer.load_state_dict(checkpoint_data["optimizer"])
        if "lr_scheduler" in checkpoint_data:
            self.lr_scheduler.lr = checkpoint_data["lr_scheduler"]["lr"]
            self.lr_scheduler.epoch = checkpoint_data["lr_scheduler"]["epoch"]

        self.epoch = epoch
        self.checkpointer.best_snapshot_fname = weight_path
        self.checkpointer.best_val_metric = metric

    def train(self):
        self.init_run(0)
        self.init_metrics()
        self.load_continuing_run()

        if self.cfg.train.continue_train and self.cfg.train.inference_only:
            return

        for self.epoch in range(self.epoch, self.cfg.train.num_epochs):
            if self.distributed:
                self.train_loader.sampler.set_epoch(self.epoch)
            if self.cfg.model.params.cfg.loss_name == "gdi_bl":
                self.model.criterion.update_alpha(self.epoch)

            self.lr_scheduler.step(self.epoch)
            self.model.train()
            train_loss = self.train_epoch()

            self.model.eval()

            val_loss, val_metric = self.val_epoch()

            if self.global_rank == 0:
                self.checkpointer.save_state(val_metric)
                self.log_writer.add_scalars(
                    'Loss', {'train': train_loss, 'val': val_loss},
                    global_step=self.epoch)
                self.log_writer.add_scalar(
                    f'Metrics/{self.key_metric}', val_metric,
                    global_step=self.epoch)

                # Logging the metrics
                if self.global_rank == 0:
                    log_out = f"[Epoch {self.epoch}] lr: {self.lr_scheduler.lr:.5f}"
                    log_out += f"--train_loss: {train_loss:.4f}"
                    log_out += f"--val_loss: {val_loss:.4f}"
                    log_out += f"--val_{self.key_metric.lower()}: {val_metric:.4f}"
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
        # losses_collector = MultiLossesCollector(local_rank=self.local_rank, cfg=self.cfg)
        # metrics_collector = SemanticSegmentationMetricsCollector(local_rank=self.local_rank, cfg=self.cfg)

        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            # check_data_loader(batch, i, "train", self.cfg)

            losses, _ = self.model(batch)
            loss = losses['loss']

            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=25, norm_type=2)
            self.optimizer.step()

            # losses_collector.update(losses)
            # metrics_collector.update(outputs=outputs)

            running_loss += loss.item()
            cur_loss = running_loss.item() / (i + 1)

            if self.global_rank == 0:
                # metrics_collector.compute()
                # cum_metrics = metrics_collector.display()
                desc = f'[{self.epoch}] Train {loss.item():.4f} / {cur_loss:.4f}'
                pbar.set_description(desc)
                # pbar.set_postfix({l:f'{cum_metrics[l]}' for l in cum_metrics})
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
            0.0, requires_grad=False).cuda(self.local_rank)
        self.metrics_collector.reset()

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                # check_data_loader(batch, i, "valid", self.cfg)

                if self.cfg.model.params.cfg.test_on_patches:
                    outputs = self.model.predict(
                        batch, inferer=self.valid_inferer)
                    self.metrics_collector.compute(
                        outputs["pred"],
                        outputs["gt"].int())

                else:
                    losses, outputs = self.model(batch)
                    loss = losses['loss']

                    running_loss.add_(loss)
                    self.metrics_collector.compute(
                        outputs["pred"],
                        outputs["gt"].int())

                if self.global_rank == 0:
                    desc = f'[{self.epoch}] Val'
                    pbar.set_description(desc)
                    pbar.update()

            if self.distributed:
                dist.all_reduce(running_loss)
                self.metrics_collector.all_reduce()

            loss_mean = running_loss / self.cfg.n_gpus / len(self.val_loader)

        if self.global_rank == 0:
            pbar.close()

        self.barrier()
        return loss_mean.item(), self.metrics_collector.mean()[self.key_metric]

    def infer(self, get_images=False, get_metrics=False):
        # Rank 0 only
        self.barrier()
        if self.global_rank != 0:
            return
        if self.cfg.data.test_metadata is None:
            self.logger.info(f"{self.cfg.data.test_metadata} is None!!")
            return

        # Setup
        metadata = self.test_metadata
        os.makedirs(self.cfg.inference_dir, exist_ok=True)
        for dataset in metadata.keys():
            os.makedirs(Path(self.cfg.inference_dir) / dataset, exist_ok=True)

        # Model
        device = self.device
        model = self.model.module if self.distributed else self.model
        state = torch.load(self.checkpointer.best_snapshot_fname)
        checkpoint_data = state["model"] if "model" in state else state
        model.load_state_dict(checkpoint_data)
        model = model.to(device)

        # Data
        test_ds = MixedTestDataset(metadata=metadata, config=self.cfg.data)
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        # Inference
        model.eval()
        inferers = {dataset_name: MySlidingInferer(
            roi_size=metadata[dataset_name].patch_size,
            input_size=(self.cfg.data.image_size, self.cfg.data.image_size),
            # input_size=None,
            sw_batch_size=8,
            overlap=0.5,
            sw_device=device,
            device=device,
        ) for dataset_name in metadata.keys()}

        if get_metrics:
            self.metrics_collector.reset()
        # Print
        self.logger.info(
            "Training finished. Running Inference on the test dataset!!")
        self.logger.info(f"Test dataset size: {len(test_ds)}")
        total_time_taken = 0

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                # check_data_loader(batch, i, "test", self.cfg)
                img_input = batch["input"].cuda(
                    self.local_rank, non_blocking=True)
                img_gt = batch["gt"].cuda(self.local_rank, non_blocking=True)
                dataset_name = batch["dataset"][0]

                if (img_input.ndim == 4) and (self.cfg.model.params.cfg.test_on_patches):
                    img_input = functional.interpolate(
                        img_input,
                        tuple(metadata[dataset_name].whole_size),
                        mode="bicubic", align_corners=True)
                    img_gt = functional.interpolate(
                        img_gt,
                        tuple(metadata[dataset_name].whole_size),
                        mode="nearest")

                batch["input"] = img_input
                batch["gt"] = img_gt
                start_time = time.time()
                logits = self.model.predict(
                    batch, inferer=inferers[dataset_name])["pred"]
                time_taken = time.time() - start_time
                total_time_taken += time_taken

                if self.cfg.model.params.cfg.arch == "SA-Unet":
                    logits = logits[:, :, 40:-41, 40:-41]
                # Crop due to aspect ratio
                if (metadata[dataset_name].crop_output is not None) and False:
                    crop_box = metadata[dataset_name].crop_output
                    logits = logits[:, :, crop_box[1]                                    :crop_box[3], crop_box[0]:crop_box[2]]

                # print(logits.shape, img_input.shape, img_gt.shape)
                # Save results
                input_image_path = batch["paths"][0][0]
                input_image_name = Path(input_image_path).stem

                torch.save(
                    logits,
                    Path(self.cfg.inference_dir) /
                    dataset_name / f"{input_image_name}.pt",
                )

                if get_metrics:
                    self.metrics_collector.compute(logits, img_gt.int())
                if get_images:
                    preds = self.metrics_collector.to_class_output(logits)
                    preds = preds.squeeze(dim=0) if preds.ndim == 4 else preds
                    out = preds[0, :, :].cpu().numpy()
                    out = (out > 0.5) * 255
                    out = np.concatenate([
                        out,
                        img_gt.squeeze().cpu().numpy().astype(np.uint8) * 255,
                    ], axis=1)

                    cv2.imwrite(str(Path(self.cfg.inference_dir) /
                                dataset_name / f"{input_image_name}.png"), out)

        if get_metrics:
            print(self.metrics_collector.mean())

        return
