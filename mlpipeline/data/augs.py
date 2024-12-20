import random
import math
from functools import wraps

import numpy as np
import cv2
import imutils
from PIL import Image
from solt.core import BaseTransform, InterpolationPropertyHolder, ImageTransform
from solt.core import DataContainer, Keypoints
from solt.utils import img_shape_checker, validate_numeric_range_parameter
from solt.constants import ALLOWED_INTERPOLATIONS

import mlpipeline.utils.common as utils
import mlpipeline.utils.generate_uncertainty_masks as mask_gen


def _maybe_process_in_chunks(process_fn, **kwargs):
    """
    Wrap OpenCV function to enable processing images with more than 4 channels.

    Limitations:
        This wrapper requires image to be the first argument and rest must be sent via named arguments.

    Args:
        process_fn: Transform function.
        kwargs: Additional parameters.

    Returns:
        numpy.ndarray: Transformed image.
    """
    @wraps(process_fn)
    def __process_fn(img: np.ndarray) -> np.ndarray:
        num_channels = img.shape[2] if len(img.shape) == 3 else 1
        if num_channels > 4:
            chunks = []
            for index in range(0, num_channels, 4):
                if num_channels - index == 2:
                    # Many OpenCV functions cannot work with 2-channel images
                    for i in range(2):
                        chunk = img[:, :, index + i: index + i + 1]
                        chunk = process_fn(chunk, **kwargs)
                        chunk = np.expand_dims(chunk, -1)
                        chunks.append(chunk)
                else:
                    chunk = img[:, :, index: index + 4]
                    chunk = process_fn(chunk, **kwargs)
                    chunks.append(chunk)
            img = np.dstack(chunks)
        else:
            img = process_fn(img, **kwargs)
        return img

    return __process_fn


def resize_layers(masks, size, use_combine=False, resample=Image.LANCZOS, threshold=1.0):
    if (masks.ndim == 3) and (masks.shape[-1] > 1):
        output_masks = []
        hard_output_mask = None

        for i in range(0, masks.shape[-1]):
            max_value = masks[:, :, i].max()

            if max_value > 200:
                output_mask = utils.resize_mask(masks[:, :, i], size)

            else:
                output_mask = masks[:, :, i]
                output_mask = utils.resize_by_pillow(
                    output_mask, size, resample=resample)
                if max_value > 1.5:
                    output_mask = np.clip(output_mask, a_min=0.0, a_max=2.0)
                else:
                    output_mask = np.clip(output_mask, a_min=0.0, a_max=1.0)
            output_masks.append(output_mask)
        output = np.stack(output_masks, axis=-1)

    else:
        max_value = masks.max()
        if max_value > 200:
            output = utils.resize_mask(masks, size)
        else:
            output = utils.resize_by_pillow(masks, size, resample=resample)
            if max_value > 1.5:
                output = np.clip(output, a_min=0.0, a_max=2.0)
            else:
                output = np.clip(output, a_min=0.0, a_max=1.0)
    return output


class Resize(BaseTransform, InterpolationPropertyHolder):
    """Transformation, which resizes the input to a given size

    Parameters
    ----------
    resize_to : tuple or int or None
        Target size ``(height_new, width_new)``.
    interpolation :
        Interpolation type.

    See also
    --------
    solt.constants.allowed_interpolations

    """

    serializable_name = "resizing"
    """How the class should be stored in the registry"""

    resample_mapping = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS,
    }

    def __init__(self, resize_to=None, interpolation="bilinear"):
        BaseTransform.__init__(self, p=1)
        InterpolationPropertyHolder.__init__(self, interpolation=interpolation)
        assert interpolation in Resize.resample_mapping.keys()

        if resize_to is not None:
            if not isinstance(resize_to, tuple) and not isinstance(resize_to, int):
                raise TypeError(
                    "The argument resize_to has an incorrect type!")
            if isinstance(resize_to, int):
                resize_to = (resize_to, resize_to)

        self.resize_to = resize_to
        self.resample = Resize.resample_mapping[interpolation]

    def _apply_img(self, img: np.ndarray, settings: dict):
        if self.resize_to is None:
            return img

        interp = ALLOWED_INTERPOLATIONS[self.interpolation[0]]
        if settings["interpolation"][1] == "strict":
            interp = ALLOWED_INTERPOLATIONS[settings["interpolation"][0]]

        output = utils.resize_by_pillow(
            img, self.resize_to[::-1], resample=self.resample)

        if output.ndim == 2:
            output = np.expand_dims(output, axis=-1)
        return output

    def _apply_mask(self, mask: np.ndarray, settings: dict):
        if self.resize_to is None:
            return mask

        output = resize_layers(mask, self.resize_to[::-1])
        return output

    def _apply_labels(self, labels, settings: dict):
        return labels

    def _apply_pts(self, pts: Keypoints, settings: dict):
        if self.resize_to is None:
            return pts
        pts_data = pts.data.copy().astype(float)

        resize_d0, resize_d1 = self.resize_to

        scale_d0 = resize_d0 / pts.frame[0]
        scale_d1 = resize_d1 / pts.frame[1]

        pts_data[:, 0] *= scale_d0
        pts_data[:, 1] *= scale_d1

        pts_data = pts_data.astype(int)
        return Keypoints(pts_data, frame=(resize_d0, resize_d1))


class RandomResizedCrop(BaseTransform, InterpolationPropertyHolder):
    """Random resized crop transform.

    Random cropping of a random size with a random aspect ratio, and then resizes the crop to a
    target size.

    Parameters
    ----------
    resize_to : tuple or int or None
        Size of the crop ``(width_new, height_new)``. If ``int``, then a square crop will be made.
    scale : tuple of flat or None
        Range of size of the origin size cropped. If None, then default Imagenet values are used -- ``(0.08, 1.0)``.
    ratio : tuple of float or None
        Range of the aspect ratio. If None, then default Imagenet values are used -- ``(3 / 4., 4 / 3.)``.

    """

    serializable_name = "random_resized_crop"
    """How the class should be stored in the registry"""

    def __init__(self, resize_to, scale=None, ratio=None, crop_size=None, interpolation="bilinear", use_combine=False, p=0.5):
        super(RandomResizedCrop, self).__init__(p=p)
        InterpolationPropertyHolder.__init__(self, interpolation=interpolation)
        assert interpolation in Resize.resample_mapping.keys()

        if scale is None:
            scale = (0.1, 1.0)
        if isinstance(scale, float):
            scale = (scale, scale)
        elif isinstance(scale, list):
            scale = tuple(scale)

        if ratio is None:
            ratio = (0.75, 1.3333333333333333)
        if isinstance(ratio, float):
            ratio = (ratio, ratio)
        elif isinstance(ratio, list):
            ratio = tuple(ratio)

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        elif isinstance(crop_size, list):
            crop_size = tuple(crop_size)

        if resize_to is not None:
            if not isinstance(resize_to, (int, tuple, list)):
                raise TypeError("Argument crop_to has an incorrect type!")

            if isinstance(resize_to, list):
                resize_to = tuple(resize_to)

            if isinstance(resize_to, tuple):
                if not isinstance(resize_to[0], int) or not isinstance(resize_to[1], int):
                    raise TypeError("Incorrect type of the crop_to!")

            if isinstance(resize_to, int):
                resize_to = (resize_to, resize_to)

        self.resize_to = resize_to
        self.scale = scale
        self.ratio = ratio
        self.crop_size = crop_size
        self.use_combine = use_combine
        self.resample = Resize.resample_mapping[interpolation]

    def sample_transform(self, data: DataContainer):
        h, w = super(RandomResizedCrop, self).sample_transform(data)
        area = h * w

        success = False
        for _attempt in range(10):
            if self.crop_size is not None:
                new_h, new_w = self.crop_size

            else:
                target_area = random.uniform(*self.scale) * area
                log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
                aspect_ratio = math.exp(random.uniform(*log_ratio))

                new_w = int(round(math.sqrt(target_area * aspect_ratio)))
                new_h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < new_w <= w and 0 < new_h <= h:
                i = random.randint(0, h - new_h)
                j = random.randint(0, w - new_w)
                success = True
                break

        if not success:
            # Fallback to central crop
            new_h, new_w = h, w
            in_ratio = w / h
            if in_ratio < min(self.ratio):
                new_w = w
                new_h = int(round(new_w / min(self.ratio)))
            elif in_ratio > max(self.ratio):
                new_h = h
                new_w = int(round(new_h * max(self.ratio)))

            i = (h - new_h) // 2
            j = (w - new_w) // 2

        self.state_dict.update({
            "crop_to": (new_w, new_h),
            "y": i,
            "x": j,
        })

    def __crop_img_or_mask(self, img_mask):
        return img_mask[
            self.state_dict["y"]: self.state_dict["y"] + self.state_dict["crop_to"][1],
            self.state_dict["x"]: self.state_dict["x"] + self.state_dict["crop_to"][0],
        ]

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        img = self.__crop_img_or_mask(img)
        if self.resize_to is None:
            return img

        interp = ALLOWED_INTERPOLATIONS[self.interpolation[0]]
        if settings["interpolation"][1] == "strict":
            interp = ALLOWED_INTERPOLATIONS[settings["interpolation"][0]]

        output = utils.resize_by_pillow(
            img, self.resize_to[::-1], resample=self.resample)

        if output.ndim == 2:
            output = np.expand_dims(output, axis=-1)
        return output

    def _apply_mask(self, mask: np.ndarray, settings: dict):
        mask = self.__crop_img_or_mask(mask)
        if self.resize_to is None:
            return mask

        output = utils.resize_mask(mask, self.resize_to[::-1])
        return output

    def _apply_labels(self, labels, settings: dict):
        return labels

    def _apply_pts(self, pts: Keypoints, settings: dict):
        raise NotImplementedError('Made for images only at this stage')


class Expand(BaseTransform):
    serializable_name = "expand"

    def __init__(self, ratio=None, p=0.5):
        super(Expand, self).__init__(p=p)

        if ratio is None:
            ratio = (1.0, 1.1)
        if isinstance(ratio, float):
            ratio = (ratio, ratio)
        elif isinstance(ratio, list):
            ratio = tuple(ratio)
        self.ratio = ratio

    def sample_transform(self, data: DataContainer):
        ratio = random.uniform(self.ratio[0], self.ratio[1])
        self.state_dict.update({
            "ratio": ratio,
        })

    @img_shape_checker
    def _apply_img(self, image: np.ndarray, settings: dict):
        height, width = image.shape[:2]
        ratio = self.state_dict["ratio"]

        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)
        self.state_dict.update({
            "left": left,
            "top": top,
        })

        if image.ndim == 3:
            expand_image = np.zeros(
                (int(height*ratio), int(width*ratio), image.shape[2]), dtype=image.dtype)
            expand_image[
                int(top): int(top + height),
                int(left): int(left + width),
                :,
            ] = image
        else:
            expand_image = np.zeros(
                (int(height*ratio), int(width*ratio)), dtype=image.dtype)
            expand_image[
                int(top): int(top + height),
                int(left): int(left + width),
            ] = image

        image = expand_image
        return image

    def _apply_mask(self, mask: np.ndarray, settings: dict):
        height, width = mask.shape[:2]
        ratio = self.state_dict["ratio"]
        left = self.state_dict["left"]
        top = self.state_dict["top"]

        if mask.ndim == 3:
            expand_mask = []
            for i in range(0, mask.shape[-1]):
                expand_mask_layer = np.full(
                    (int(height*ratio), int(width*ratio)), dtype=mask.dtype, fill_value=0.0)
                expand_mask_layer[
                    int(top): int(top + height),
                    int(left): int(left + width)
                ] = mask[:, :, i]
                expand_mask.append(expand_mask_layer)
            expand_mask = np.stack(expand_mask, axis=-1)
        else:
            expand_mask = np.full(
                (int(height*ratio), int(width*ratio)), dtype=mask.dtype, fill_value=0.0)
            expand_mask[
                int(top): int(top + height),
                int(left): int(left + width),
            ] = mask

        mask = expand_mask
        return mask

    def _apply_labels(self, labels, settings: dict):
        return labels

    def _apply_pts(self, pts: Keypoints, settings: dict):
        raise NotImplementedError('Made for images only at this stage')


class CLAHE(BaseTransform):
    """Apply Contrast Limited Adaptive Histogram Equalization to the input image.

    Parameters
    ----------
    clip_limit : float or [float, float]
    tile_grid_size : [int, int]
    p : float
    """

    serializable_name = "clahe"
    """How the class should be stored in the registry"""

    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8), p=0.5):
        super(CLAHE, self).__init__(p=p)
        if clip_limit is None:
            clip_limit = (1.0, 4.0)
        if isinstance(clip_limit, float):
            assert clip_limit >= 1.0
            clip_limit = (1.0, clip_limit)
        elif isinstance(clip_limit, list):
            clip_limit = tuple(clip_limit)

        if tile_grid_size is None:
            tile_grid_size = (8, 8)
        if isinstance(tile_grid_size, int):
            tile_grid_size = (tile_grid_size, tile_grid_size)
        elif isinstance(tile_grid_size, list):
            tile_grid_size = tuple(tile_grid_size)

        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def sample_transform(self, data: DataContainer):
        clip_limit = random.uniform(self.clip_limit[0], self.clip_limit[1])
        self.state_dict.update({"clip_limit": clip_limit})

    def _clahe(self, img, clip_limit):
        if img.dtype != np.uint8:
            raise TypeError("clahe supports only uint8 inputs")

        clahe_mat = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=self.tile_grid_size)

        if len(img.shape) == 2 or img.shape[2] == 1:
            img = clahe_mat.apply(img)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            img[:, :, 0] = clahe_mat.apply(img[:, :, 0])
            img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

        return img

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        img = self._clahe(img, self.state_dict["clip_limit"])
        return img

    def _apply_mask(self, mask: np.ndarray, settings: dict):
        return mask

    def _apply_labels(self, labels, settings: dict):
        return labels

    def _apply_pts(self, pts: Keypoints, settings: dict):
        raise NotImplementedError('Made for images only at this stage')


class RandomToneCurve(BaseTransform):
    """
    Randomly change the relationship between bright and dark areas of the image
    by manipulating its tone curve.

    Parameters
    ----------
    scale : float
    p : float
    """

    serializable_name = "random_tone_curve"
    """How the class should be stored in the registry"""

    def __init__(self, scale=0.1, p=0.5):
        super(RandomToneCurve, self).__init__(p=p)
        self.scale = scale

    def sample_transform(self, data: DataContainer):
        self.state_dict.update({
            "low_y": np.clip(np.random.normal(loc=0.25, scale=self.scale), 0, 1),
            "high_y": np.clip(np.random.normal(loc=0.75, scale=self.scale), 0, 1),
        })

    def _move_tone_curve(self, img, low_y, high_y):
        input_dtype = img.dtype

        if low_y < 0 or low_y > 1:
            raise ValueError("low_shift must be in range [0, 1]")
        if high_y < 0 or high_y > 1:
            raise ValueError("high_shift must be in range [0, 1]")

        if input_dtype != np.uint8:
            raise ValueError("Unsupported image type {}".format(input_dtype))

        t = np.linspace(0.0, 1.0, 256)

        # Defines responze of a four-point bezier curve
        def evaluate_bez(t):
            return 3 * (1 - t) ** 2 * t * low_y + 3 * (1 - t) * t**2 * high_y + t**3

        evaluate_bez = np.vectorize(evaluate_bez)
        remapping = np.rint(evaluate_bez(t) * 255).astype(np.uint8)

        lut_fn = _maybe_process_in_chunks(cv2.LUT, lut=remapping)
        img = lut_fn(img)
        return img

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        if img.ndim == 2:
            img = self._move_tone_curve(
                img,
                self.state_dict["low_y"],
                self.state_dict["high_y"],
            )
        else:
            for i in range(0, img.shape[-1]):
                img[..., i] = self._move_tone_curve(
                    img[..., i],
                    self.state_dict["low_y"],
                    self.state_dict["high_y"],
                )
        return img

    def _apply_mask(self, mask: np.ndarray, settings: dict):
        return mask

    def _apply_labels(self, labels, settings: dict):
        return labels

    def _apply_pts(self, pts: Keypoints, settings: dict):
        raise NotImplementedError('Made for images only at this stage')


class CutOut(ImageTransform):
    """Does cutout augmentation.

    Parameters
    ----------
    cutout_size : tuple or int or float or None
        The size of the cutout. If None, then it is equal to 2.
    num_cutout
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.
    p : float
        Probability of applying this transform.
    """

    serializable_name = "my_cutout"
    """How the class should be stored in the registry"""

    def __init__(self, cutout_size=2, num_cutout=1, data_indices=None, p=0.5):
        super(CutOut, self).__init__(p=p, data_indices=data_indices)
        if not isinstance(cutout_size, (int, tuple, list, float)):
            raise TypeError("Cutout size is of an incorrect type!")

        if isinstance(cutout_size, list):
            cutout_size = tuple(cutout_size)

        if isinstance(cutout_size, tuple):
            if not isinstance(cutout_size[0], (int, float)) or not isinstance(cutout_size[1], (int, float)):
                raise TypeError

        if isinstance(cutout_size, (int, float)):
            cutout_size = (cutout_size, cutout_size)
        if not isinstance(cutout_size[0], type(cutout_size[1])):
            raise TypeError("CutOut sizes must be of the same type")

        self.cutout_size = cutout_size
        self.num_cutout = num_cutout

    # TODO: refactor from OpenCV (w, h)/(_x, _y) to new (h, w, ...)/(d0, d1, ...)
    def sample_transform(self, data: DataContainer):
        for index in range(0, self.num_cutout):
            h, w = super(CutOut, self).sample_transform(data)[:2]
            if isinstance(self.cutout_size[0], float):
                cut_size_x = int(self.cutout_size[0] * w)
            else:
                cut_size_x = self.cutout_size[0]

            if isinstance(self.cutout_size[1], float):
                cut_size_y = int(self.cutout_size[1] * h)
            else:
                cut_size_y = self.cutout_size[1]

            if cut_size_x > w or cut_size_y > h:
                raise ValueError("Cutout size is too large!")

            self.state_dict[f"x_{index}"] = int(
                random.random() * (w - cut_size_x))
            self.state_dict[f"y_{index}"] = int(
                random.random() * (h - cut_size_y))
            self.state_dict[f"cut_size_x_{index}"] = cut_size_x
            self.state_dict[f"cut_size_y_{index}"] = cut_size_y
        return

    def __cutout_img(self, img):
        for index in range(0, self.num_cutout):
            state = self.state_dict
            img[
                state[f"y_{index}"]: state[f"y_{index}"] + state[f"cut_size_y_{index}"],
                state[f"x_{index}"]: state[f"x_{index}"] + state[f"cut_size_x_{index}"],
            ] = 0
        return img

    def _apply_img(self, img: np.ndarray, settings: dict):
        return self.__cutout_img(img)


class Contrast(ImageTransform):
    """Transform randomly changes the contrast

    Parameters
    ----------
    p : float
        Probability of applying this transform,
    contrast_range : tuple or float or None
        Gain of the noise. Indicates percentage of indices, which will be changed.
        If float, then ``gain_range = (1-contrast_range, 1+contrast_range)``.
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.

    """

    _default_range = (1, 1)
    serializable_name = "my_contrast"
    """How the class should be stored in the registry"""

    def __init__(self, p=0.5, contrast_range=0.1, brightness=0, data_indices=None):
        super(Contrast, self).__init__(p=p, data_indices=data_indices)

        if isinstance(contrast_range, float):
            contrast_range = (1 - contrast_range, 1 + contrast_range)
        if isinstance(brightness, int):
            brightness = (brightness, brightness)

        self.contrast_range = validate_numeric_range_parameter(
            contrast_range, self._default_range, 0)
        self.brightness = brightness

    def sample_transform(self, data):
        contrast_mul = random.uniform(
            self.contrast_range[0], self.contrast_range[1])
        lut = np.arange(0, 256) * contrast_mul
        lut = np.clip(lut, 0, 255).astype("uint8")
        brightness = random.uniform(self.brightness[0], self.brightness[1])
        self.state_dict = {"contrast_mul": contrast_mul,
                           "LUT": lut, "brightness": brightness}

    def _apply_img(self, img: np.ndarray, settings: dict):
        mean_value_before = int(img.mean())
        transformed = cv2.LUT(img, self.state_dict["LUT"])
        mean_value_after = int(transformed.mean())

        sum_image = np.sum(img.astype(int), axis=-1)
        threshold = 6 if img.shape[-1] == 1 else 20
        fov = (sum_image > threshold).astype(bool)
        transformed = np.where(
            np.stack([fov] * img.shape[-1], axis=-1),
            transformed + (mean_value_before - mean_value_after) +
            self.state_dict["brightness"],
            transformed,
        )

        transformed = np.clip(transformed, 0, 255).astype("uint8")
        return transformed


class UncertaintyMask(BaseTransform):
    """
    Parameters
    ----------
    """

    serializable_name = "uncertainty_mask"
    """How the class should be stored in the registry"""

    def __init__(self, label_name="t", transform_function=None, r=1):
        super(UncertaintyMask, self).__init__(p=1.0)
        self.label_name = label_name
        self.transform_function = transform_function
        self.r = r

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        return img

    def _apply_mask(self, mask: np.ndarray, settings: dict):
        assert len(np.unique(mask)) <= 2, np.unique(mask)
        assert np.max(mask) in [0, 255], np.max(mask)
        if mask.ndim == 3:
            mask = np.squeeze(mask, axis=-1)

        boundary_mask = mask_gen.extract_boundary_uncertainty_map(
            mask, self.transform_function)[0]
        thickness_mask = mask_gen.extract_thickness_uncertainty_map(
            mask,
            tr=self.transform_function,
            target_c_label=self.label_name,
            kernel_ratio=self.r,
        )[0]
        combined_mask = mask_gen.extract_combined_uncertainty_map(
            boundary_mask, thickness_mask,
            self.label_name,
        )

        output = np.stack([mask, combined_mask], axis=-1)
        return output

    def _apply_labels(self, labels, settings: dict):
        return labels

    def _apply_pts(self, pts: Keypoints, settings: dict):
        raise NotImplementedError('Made for images only at this stage')
