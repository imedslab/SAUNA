import os
import importlib
import socket
import random
import copy
import logging as log
import pickle
import warnings
from time import localtime, strftime
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf
from PIL import Image


def init_obj_cls(string_def):
    string_parts = string_def.split(".")
    obj_cls = getattr(importlib.import_module(
        ".".join(string_parts[:-1])), string_parts[-1])
    return obj_cls


def init_obj(string_def, params):
    obj_cls = init_obj_cls(string_def)
    if params is None:
        params = {}
    return obj_cls(**params)


def find_free_port():
    s = socket.socket()
    s.bind(("", 0))  # Bind to a free port provided by the host.
    return s.getsockname()[1]


def init_dist_url(host):
    ip = socket.gethostbyname(host)
    port = find_free_port()
    return "tcp://{}:{}".format(ip, port)


def init_random(seed, rank):
    torch.cuda.set_device(rank)
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    torch.manual_seed(seed + rank)
    cudnn.benchmark = True


def init_conf(cwd, conf_path):
    OmegaConf.register_resolver(
        "now", lambda pattern: strftime(pattern, localtime()))
    config = OmegaConf.load(str(conf_path))
    conf_cli = OmegaConf.from_cli()

    for entry in config.defaults:
        assert len(entry) == 1
        for k, v in entry.items():
            if k in conf_cli:
                v = conf_cli[k]
            entry_path = cwd / "config" / k / f"{v}.yaml"
            entry_conf = OmegaConf.load(str(entry_path))
            config = OmegaConf.merge(config, entry_conf)

    cfg = OmegaConf.merge(config, conf_cli)
    cfg.original_cwd = str(cwd)
    cfg.snapshot_dir = str(cwd / cfg.snapshot_dir)

    write_config_to_snapshot(cfg)
    return cfg


def write_config_to_snapshot(cfg):
    snapshot_dir = Path(cfg.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    with open(snapshot_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)
    return


def create_random_masks(bs, cfg, model, n_batches=None):
    n_scalars = cfg.input.map['IMG']['pos'] - 1
    if n_batches is None:
        probs = np.random.rand(bs, model.n_inputs)
        mask_scalar = probs[:, :n_scalars] > cfg.mask_scalar_rate
        mask_img = probs[:, n_scalars:] > cfg.mask_img_rate
        masks = torch.tensor(np.concatenate(
            (mask_scalar, mask_img), -1)).unsqueeze(-1)
    else:
        probs = np.random.rand(n_batches, bs, model.n_inputs)
        mask_scalar = probs[:, :, :n_scalars] > cfg.mask_scalar_rate
        mask_img = probs[:, :, n_scalars:] > cfg.mask_img_rate
        masks = torch.tensor(np.concatenate(
            (mask_scalar, mask_img), -1)).unsqueeze(-1)
    return masks


def calculate_metric(metric_func, y_true, y_pred, **kwargs):
    result = None
    if len(y_pred) == len(y_true) and len(y_pred) > 0:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = metric_func(y_true, y_pred, **kwargs)
        except ValueError:
            pass
    return result


def count_parameters(model):
    n_trainable_params = sum(p.numel()
                             for p in model.parameters() if p.requires_grad)
    n_params = sum(p.numel() for p in model.parameters())
    return n_trainable_params, n_params


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        if tensor.device.type == 'cuda':
            return tensor.cpu().detach().numpy()
        else:
            return tensor.detach().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError(
            "Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError(
            "Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray


def read_image(image_path, gray=False):
    ext = Path(image_path).suffix.lower()

    if ext in [".jpg", ".jpeg", ".png"]:
        image = cv2.imread(image_path)
    elif ext in [".tif", ".gif", ".ppm"]:
        image = np.array(Image.open(image_path).convert("RGB"))[
            :, :, [2, 1, 0]]

    if gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def remove_empty_img_rows(root, df):
    rows = []
    for ind, row in df.iterrows():
        img_filename = create_img_name(row)
        img_fullname = os.path.join(root, img_filename)
        if os.path.isfile(img_fullname):
            rows.append(row)
        else:
            print(f'Not found {img_fullname}.')
    return pd.DataFrame(rows)


def create_img_name(entry):
    img_filename = f"{entry['ID']}_{int(entry['visit']):02d}_{entry['Side']}.png"
    return img_filename


def post_process_data(splitter, proc_targets):
    split_data = []
    for fold_i, (train, val) in enumerate(splitter):
        if proc_targets is not None:
            train = proc_targets(train)
            val = proc_targets(val)

        split_data.append((train, val))
    split_data = tuple(split_data)
    return split_data


def to_cpu(x, required_grad=False, use_numpy=True):
    x_cpu = x

    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            if use_numpy:
                x_cpu = x.to('cpu').detach().numpy()
            elif required_grad:
                x_cpu = x.to('cpu')
            else:
                x_cpu = x.to('cpu').required_grad_(False)
        elif use_numpy:
            if x.requires_grad:
                x_cpu = x.detach().numpy()
            else:
                x_cpu = x.numpy()

    return x_cpu


def mask_list(l, masks):
    return [l[i].item() if isinstance(l, torch.Tensor) else l[i] for i in range(masks.shape[0]) if masks[i]]


def compute_probs(x, tau=1.0, dim=-1, numpy=True):
    tau = tau if tau is not None else 1.0

    probs = torch.softmax(x * tau, dim=dim)

    if numpy:
        probs = to_numpy(probs)
    return probs


def resize_by_pillow(image, size, resample):
    unsqueeze_needed = False
    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.squeeze(image, axis=-1)
        unsqueeze_needed = True

    image = Image.fromarray(image).resize(size, resample=resample)
    image = np.array(image)

    if unsqueeze_needed:
        image = np.expand_dims(image, axis=-1)
    return image


def resize_mask(image, size, resample=Image.NEAREST):
    image = resize_by_pillow(image, size, resample=resample)
    image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)[1]
    return image


def convert_grayscale_to_heatmap(image, raw_image=None, colormap="inferno", show_path=None):
    colormap = plt.get_cmap(colormap)

    if (show_path is not None) and (raw_image is not None):
        fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
        fig_img = plt.imshow(raw_image, interpolation="nearest", cmap=colormap)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(show_path)

    heatmap = (colormap(image) * 255).astype(np.uint8)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    return heatmap
