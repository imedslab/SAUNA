from typing import Tuple, Union

import os
import itertools
import random
from pathlib import Path

import numpy as np
import cv2
import tqdm
from natsort import natsorted
from PIL import Image
from skimage import morphology

import mlpipeline.utils.common as utils


def generate_sequential_patches(
    root_dir, patches_dir,
    input_set_name, gt_set_names,
    input_ext="png", gt_ext="png",
    whole_resize=None,
    patch_size: Union[Tuple[int], int] = 256,
    step_size: Union[Tuple[int], int] = 128,
    start_index: int = 0,
    get_top: bool = False,
    top_k: int = 0,
    use_raw_index: bool = False,
):
    """
    Generate sequential patches of images from FIVES.
    """
    if not isinstance(gt_set_names, list):
        gt_set_names = [gt_set_names]

    image_paths = natsorted(
        (Path(root_dir) / input_set_name).glob(f"*.{input_ext}"))
    print(len(image_paths))

    os.makedirs(f"{patches_dir}/{input_set_name}", exist_ok=True)
    for gt_set_name in gt_set_names:
        os.makedirs(f"{patches_dir}/{gt_set_name}", exist_ok=True)

    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    if isinstance(step_size, int):
        step_size = (step_size, step_size)
    if not get_top:
        top_k = 0

    for image_path in image_paths:
        input_name = image_path.stem
        image = np.array(Image.open(str(image_path)))

        # gt_name = name
        gt_name = input_name.replace("_training", "_manual")
        gt_paths = [Path(root_dir) / gt_set_name /
                    f"{gt_name}.{gt_ext}" for gt_set_name in gt_set_names]
        gts = [np.array(Image.open(str(gt_path)).convert("L"))
               for gt_path in gt_paths]

        # resize if asked
        if whole_resize is not None:
            image = cv2.resize(
                image, (whole_resize, whole_resize), interpolation=cv2.INTER_LINEAR)
            gts = [cv2.resize(gt, (whole_resize, whole_resize),
                              interpolation=cv2.INTER_NEAREST) for gt in gts]

        # make grids
        grids = itertools.product(
            range(0, image.shape[0], step_size[1]),
            range(0, image.shape[1], step_size[0]),
        )
        patches = []
        index = start_index

        for (y_start, x_start) in grids:
            if (y_start + patch_size[1] > image.shape[0]) or (x_start + patch_size[0] > image.shape[1]):
                continue

            patch = image[y_start: (y_start + patch_size[1]),
                          x_start: (x_start + patch_size[0])]
            gt_patches = [gt[y_start: (
                y_start + patch_size[1]), x_start: (x_start + patch_size[0])] for gt in gts]
            num_pos = np.sum(gt_patches[0] > 120)

            if get_top:
                # Count number of positive pixels in patch
                patches.append([index, patch, gt_patches, num_pos])

            elif num_pos > 0:
                Image.fromarray(patch).save(
                    str(Path(patches_dir) / input_set_name / f"{input_name}_{index:05d}.{input_ext}"))
                for gt_patch, gt_set_name in zip(gt_patches, gt_set_names):
                    Image.fromarray(gt_patch).save(
                        str(Path(patches_dir) / gt_set_name / f"{gt_name}_{index:05d}.{gt_ext}"))

            if use_raw_index or num_pos > 0:
                index += 1

        if not get_top:
            continue

        # Get patches with most positive pixels only
        patches = sorted(patches, key=lambda x: x[3], reverse=True)
        top_to_median = range(0, len(patches) // 2)
        top_indices = random.sample(top_to_median, k=top_k)
        patches = [patches[index] for index in top_indices]

        for j, (index, patch, gt_patches, num_pos) in enumerate(patches, start=start_index):
            i = index if use_raw_index else j
            Image.fromarray(patch).save(
                str(Path(patches_dir) / input_set_name / f"{input_name}_{i:05d}.{input_ext}"))
            for gt_patch, gt_set_name in zip(gt_patches, gt_set_names):
                Image.fromarray(gt_patch).save(
                    str(Path(patches_dir) / gt_set_name / f"{gt_name}_{i:05d}.{gt_ext}"))
    return


def generate_training_patches_multi_step():
    train_dir = "./FIVES/train"
    patches_dir = "./FIVES/train_patches_512"
    input_set_name = "Original"
    gt_set_name = "GroundTruth"

    generate_sequential_patches(
        train_dir, patches_dir,
        input_set_name, gt_set_name,
        patch_size=512, step_size=256, start_index=0)
    generate_sequential_patches(
        train_dir, patches_dir,
        input_set_name, gt_set_name,
        patch_size=1024, step_size=256, start_index=49)
    generate_sequential_patches(
        train_dir, patches_dir,
        input_set_name, gt_set_name,
        patch_size=256, step_size=128, start_index=74,
        get_top=True, top_k=25, use_raw_index=False)
    generate_sequential_patches(
        train_dir, patches_dir,
        input_set_name, gt_set_name,
        patch_size=(1024, 768), step_size=256, start_index=99)
    return


def generate_skeleton():
    gt_dir = "./FIVES/test/GroundTruth"
    output_dir = "./FIVES/test/Skeleton"
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    image_paths = natsorted(Path(gt_dir).glob("*.png"))
    print(len(image_paths))
    pbar = tqdm.tqdm(total=len(image_paths))
    os.makedirs(output_dir, exist_ok=True)

    for index, image_path in enumerate(image_paths):
        name = image_path.stem
        label_mask = utils.read_image(str(image_path), gray=True)

        skel = morphology.skeletonize(
            (label_mask / 255.0).round()).astype(np.uint8)
        dilated = morphology.dilation(skel, kernel) * 255
        cv2.imwrite(str(Path(output_dir) / f"{name}.png"), dilated)

        pbar.set_description(str(index))
        pbar.update()
    return


if __name__ == "__main__":
    generate_training_patches_multi_step()
