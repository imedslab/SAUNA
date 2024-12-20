import os
import re
import json
import csv
import math
import itertools
from pathlib import Path

import click
import numpy as np
import cv2
import torch
import torch.nn.functional as functional
from omegaconf import OmegaConf
from natsort import natsorted

import mlpipeline.utils.common as common
from mlpipeline.metrics.metric_collectors import SemanticSegmentationMetricsCollector


def resize_output_tensor(output_tensor, size):
    output_tensor = functional.interpolate(
        output_tensor,
        size,
        mode="bicubic",
        align_corners=False,
    )
    return output_tensor


class Evaluator:
    def __init__(
        self,
        output_dir, log_dir, visual_dir,
        metadata,
        cfg,
        dataset_name,
        seeds, num_folds,
        key_metric="F1",
        resize_gt=False,
    ):
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.visual_dir = str(Path(visual_dir) / dataset_name)
        self.metadata = metadata
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.seeds = seeds
        self.num_folds = num_folds
        self.resize_gt = resize_gt

        self.metrics_collector = SemanticSegmentationMetricsCollector(
            local_rank=0,
            cfg=self.cfg,
        )
        self.key_metric = key_metric
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        assert key_metric in SemanticSegmentationMetricsCollector.METRIC_NAMES

        # Get all directories of inference results
        self.experiment_dirs = Path(output_dir).glob(f"*")
        self.experiment_dirs = [d for d in self.experiment_dirs if d.is_dir()]
        print(output_dir, len(self.experiment_dirs))

        # Get gt files
        gt_dir = Path(cfg.data.image_dir.base) / metadata[dataset_name].gt_dir
        self.gt_paths = natsorted(gt_dir.glob(
            f"*.{metadata[dataset_name]['gt_ext']}"))
        self.gt_paths = [gt_path for gt_path in self.gt_paths]
        self.gt_masks = []

        for gt_path in self.gt_paths:
            # Read gt data
            gt_mask = common.read_image(str(gt_path), gray=True)
            self.target_size = (gt_mask.shape[0], gt_mask.shape[1])
            gt_mask = torch.tensor(
                gt_mask / 255.0).long().unsqueeze(dim=0).to(self.device)
            self.gt_masks.append(gt_mask)

        print("GT:", len(self.gt_masks))

    def _map_gt_to_input_name(self, gt_name):
        if self.dataset_name == "CHASEDB1":
            return gt_name.replace("_1stHO", "")
        elif self.dataset_name == "DRIVE":
            return gt_name.replace("_manual1", "_test")
        elif self.dataset_name == "STARE":
            return gt_name.replace(".ah", "")
        return gt_name

    def visualize_results(self, gt_path, output_logits, setting, seed):
        image_name = self._map_gt_to_input_name(gt_path.stem)

        preds = self.metrics_collector.to_class_output(
            torch.tensor(output_logits), dim=1).squeeze(dim=1)
        preds = preds.cpu().numpy().transpose(1, 2, 0)
        preds = (preds > self.cfg.metrics.threshold)
        preds = (preds * 255).astype(np.uint8).squeeze(axis=-1)

        gt_image = common.read_image(str(gt_path), gray=True)
        if gt_image.shape[0] != preds.shape[0] or gt_image.shape[1] != preds.shape[1]:
            preds = cv2.resize(preds, (gt_image.shape[1], gt_image.shape[0]))

        cv2.imwrite(
            str(Path(self.visual_dir) / setting /
                f"seed-{seed}_name-{image_name}.png"),
            np.concatenate([preds, gt_image], axis=1),
        )

    def compute_metrics_on_setting(self, setting):
        metrics_dict = dict()

        # Filter by setting
        experiment_dirs = [
            d for d in self.experiment_dirs
            if d.name.startswith(setting)
        ]
        os.makedirs(Path(self.visual_dir) / setting, exist_ok=True)

        # Loop over seeds
        for seed in self.seeds:
            output_masks = []
            self.metrics_collector.reset()

            for gt_path in self.gt_paths:
                image_name = self._map_gt_to_input_name(gt_path.stem)
                # Get output data, and average over folds
                fold_outputs = []

                for fold_index in self.num_folds:
                    seed_str = f"seed:{seed}"
                    fold_str = f"fold:{fold_index}"
                    experiment_dir = [
                        d for d in experiment_dirs
                        if (seed_str in str(d)) and (fold_str in str(d))
                    ]

                    if len(experiment_dir) == 1:
                        experiment_dir = experiment_dir[0]
                    else:
                        print(
                            f"Seed {seed} and Fold {fold_index} not found: {len(experiment_dir)}\n")
                        continue

                    # Read output data
                    output_path = experiment_dir / \
                        self.dataset_name / f"{image_name}.pt"
                    output_logits = torch.load(
                        output_path, map_location=self.device)
                    # output_logits = output_logits.squeeze(dim=0)
                    output_logits = resize_output_tensor(
                        output_logits, self.target_size)
                    # Collect
                    fold_outputs.append(output_logits)

                # Get average over folds
                if len(fold_outputs) > 1:
                    seed_output = torch.mean(
                        torch.stack(fold_outputs, dim=0), dim=0)
                else:
                    seed_output = fold_outputs[0]

                # seed_output = self.metrics_collector.to_class_output(seed_output)
                self.visualize_results(gt_path, seed_output, setting, seed)
                output_masks.append(seed_output)

            # Compute metrics
            # output_masks = torch.cat(output_masks, dim=0)
            # gt_masks = torch.cat(self.gt_masks, dim=0)
            print(len(output_masks))
            assert len(output_masks) > 0
            self.metrics_collector.compute(output_masks, self.gt_masks)
            # self.metrics_collector.compute(output_masks, gt_masks)
            metrics_dict[seed] = self.metrics_collector.mean()

        # Compute statistics over runs
        summary_dict = {"mean": dict(), "std": dict()}
        for metric_name in SemanticSegmentationMetricsCollector.METRIC_NAMES:
            values = torch.tensor([metrics_dict[seed][metric_name]
                                  for seed in metrics_dict.keys()])
            summary_dict["mean"][metric_name] = torch.mean(values).item()
            summary_dict["std"][metric_name] = torch.std(
                values, unbiased=False).item()

        return summary_dict

    def run(self):
        os.makedirs(self.visual_dir, exist_ok=True)
        # Get list of hyperparam settings, independent of seed and fold
        settings = set()

        for experiment_dir in self.experiment_dirs:
            basename = experiment_dir.name
            seed_index = basename.find("_seed:")
            if seed_index < 0:
                continue

            setting = basename[:seed_index]
            settings.add(setting)

        self.settings = list(settings)
        print(f"Settings: {len(self.settings)}", self.settings[:2])
        eval_results = []

        # Loop over settings
        for setting in self.settings:
            print(setting)
            with torch.no_grad():
                metrics = self.compute_metrics_on_setting(setting)
            eval_results.append([setting, metrics])
            print(metrics)
            print("\n")

        # Log best result
        best_result = max(
            eval_results, key=lambda x: x[1]["mean"][self.key_metric])
        method_name = re.findall(f"_method:(.*?)_", best_result[0])[0]
        reduction = self.metrics_collector.reduction
        json.dump(
            {
                "method": method_name,
                "dataset": self.dataset_name,
                "setting": best_result[0],
                "mean": best_result[1]["mean"],
                "std": best_result[1]["std"],
            },
            open(Path(self.log_dir) /
                 f"{self.dataset_name}_{reduction}.json", "w+"),
            indent=4)
        return


@click.command()
@click.option("--config")
@click.option("--output_dir")
@click.option("--log_dir")
@click.option("--visual_dir")
@click.option("--metadata_path")
@click.option("--dataset_name")
@click.option("--seeds", default="99999", type=str)
@click.option("--num_folds", default="0")
def main(
    config: str,
    output_dir: str, log_dir: str, visual_dir: str,
    metadata_path: str,
    dataset_name: str,
    seeds: str, num_folds: str,
):
    # Get config
    cwd = Path().cwd()
    conf_path = cwd / "config" / "experiment" / f"{config}.yaml"
    cfg = OmegaConf.load(str(conf_path))
    metadata = OmegaConf.load(metadata_path)

    seeds = [int(x) for x in seeds.split(",")]
    num_folds = [int(x) for x in num_folds.split(",")]

    evaluator = Evaluator(
        output_dir=output_dir,
        log_dir=log_dir,
        visual_dir=visual_dir,
        metadata=metadata,
        cfg=cfg,
        dataset_name=dataset_name,
        seeds=seeds,
        num_folds=num_folds,
    )
    evaluator.run()


if __name__ == "__main__":
    main()
