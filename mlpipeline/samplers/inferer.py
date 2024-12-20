from typing import List, Dict, Tuple, Union, Sequence, Callable, Mapping, Any

import warnings

import tqdm
import torch
import torch.nn.functional as functional
import monai
from monai.data.meta_tensor import MetaTensor
from monai.transforms import Resize
from monai.utils import BlendMode, PytorchPadMode, ensure_tuple, fall_back_tuple, look_up_option
from monai.inferers.utils import compute_importance_map, _get_scan_interval, convert_data_type
from monai.data.utils import dense_patch_slices, get_valid_patch_size


def sliding_window_inference(
    inputs: torch.Tensor,
    roi_size: Union[Sequence[int], int],
    input_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor], Dict[Any, torch.Tensor]]],
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    progress: bool = False,
    roi_weight_map: Union[torch.Tensor, None] = None,
    *args: Any,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[Any, torch.Tensor]]:
    """Use SlidingWindow method to execute inference.
    Note:
        must be channel first, support both 2D and 3D.
        input data must have batch dim.
        execute on 1 image/per inference, run a batch of window slices of 1 input image.
    """
    compute_dtype = inputs.dtype
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise ValueError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    batch_size, _, *image_size_ = inputs.shape

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i])
                       for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = functional.pad(inputs, pad=pad_size, mode=look_up_option(
        padding_mode, PytorchPadMode), value=cval)

    scan_interval = _get_scan_interval(
        image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    if valid_patch_size == roi_size and (roi_weight_map is not None):
        importance_map = roi_weight_map
    else:
        try:
            importance_map = compute_importance_map(
                valid_patch_size, mode=mode, sigma_scale=sigma_scale, device=device)
        except BaseException as e:
            raise RuntimeError(
                "Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'."
            ) from e
    importance_map = convert_data_type(
        importance_map, torch.Tensor, device, compute_dtype)[0]  # type: ignore
    # handle non-positive weights
    min_non_zero = max(importance_map[importance_map != 0].min().item(), 1e-3)
    importance_map = torch.clamp(importance_map.to(
        torch.float32), min=min_non_zero).to(compute_dtype)

    # Perform predictions
    dict_key, output_image_list, count_map_list = None, [], []
    _initialized_ss = -1
    # whether the predictor's output is a tensor (instead of dict/tuple)
    is_tensor_output = True

    # for each patch
    for slice_g in tqdm(range(0, total_slices, sw_batch_size)) if progress else range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(
            slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1),
             slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat(
            [convert_data_type(inputs[win_slice], torch.Tensor)[0]
             for win_slice in unravel_slice]
        ).to(sw_device)

        # Resize
        assert window_data.shape[2] == roi_size[1], window_data.shape
        assert window_data.shape[3] == roi_size[0], window_data.shape
        if (window_data.shape[2] != input_size[1]) or (window_data.shape[3] != input_size[0]):
            window_data = functional.interpolate(
                window_data,
                input_size,
                mode="bicubic",
                align_corners=False,
            )
        # Run
        # batched patch segmentation
        seg_prob_out = predictor(window_data, *args, **kwargs)
        # Resize
        if (roi_size[0] != input_size[0]) or (roi_size[1] != input_size[1]):
            seg_prob_out = functional.interpolate(
                seg_prob_out,
                roi_size,
                mode="bicubic",
                align_corners=False,
            )
            window_data = functional.interpolate(
                window_data,
                roi_size,
                mode="bicubic",
                align_corners=False,
            )

        # convert seg_prob_out to tuple seg_prob_tuple, this does not allocate new memory.
        seg_prob_tuple: Tuple[torch.Tensor, ...]
        if isinstance(seg_prob_out, torch.Tensor):
            seg_prob_tuple = (seg_prob_out,)
        elif isinstance(seg_prob_out, Mapping):
            if dict_key is None:
                # track predictor's output keys
                dict_key = sorted(seg_prob_out.keys())
            seg_prob_tuple = tuple(seg_prob_out[k] for k in dict_key)
            is_tensor_output = False
        else:
            seg_prob_tuple = ensure_tuple(seg_prob_out)
            is_tensor_output = False

        # for each output in multi-output list
        for ss, seg_prob in enumerate(seg_prob_tuple):
            seg_prob = seg_prob.to(device)  # BxCxMxNxP or BxCxMxN

            # compute zoom scale: out_roi_size/in_roi_size
            zoom_scale = []
            for axis, (img_s_i, out_w_i, in_w_i) in enumerate(
                zip(image_size, seg_prob.shape[2:], window_data.shape[2:])
            ):
                _scale = out_w_i / float(in_w_i)
                if not (img_s_i * _scale).is_integer():
                    warnings.warn(
                        f"For spatial axis: {axis}, output[{ss}] will have non-integer shape. Spatial "
                        f"zoom_scale between output[{ss}] and input is {_scale}. Please pad inputs."
                    )
                zoom_scale.append(_scale)

            if _initialized_ss < ss:  # init. the ss-th buffer at the first iteration
                # construct multi-resolution outputs
                output_classes = seg_prob.shape[1]
                output_shape = [batch_size, output_classes] + [
                    int(image_size_d * zoom_scale_d) for image_size_d, zoom_scale_d in zip(image_size, zoom_scale)
                ]
                # allocate memory to store the full output and the count for overlapping parts
                output_image_list.append(torch.zeros(
                    output_shape, dtype=compute_dtype, device=device))
                count_map_list.append(torch.zeros(
                    [1, 1] + output_shape[2:], dtype=compute_dtype, device=device))
                _initialized_ss += 1

            # resizing the importance_map
            resizer = Resize(
                spatial_size=seg_prob.shape[2:], mode="nearest", anti_aliasing=False)

            # store the result in the proper location of the full output. Apply weights from importance map.
            for idx, original_idx in zip(slice_range, unravel_slice):
                # zoom roi
                # 4D for 2D image, 5D for 3D image
                original_idx_zoom = list(original_idx)
                for axis in range(2, len(original_idx_zoom)):
                    zoomed_start = original_idx[axis].start * \
                        zoom_scale[axis - 2]
                    zoomed_end = original_idx[axis].stop * zoom_scale[axis - 2]
                    if not zoomed_start.is_integer() or (not zoomed_end.is_integer()):
                        warnings.warn(
                            f"For axis-{axis-2} of output[{ss}], the output roi range is not int. "
                            f"Input roi range is ({original_idx[axis].start}, {original_idx[axis].stop}). "
                            f"Spatial zoom_scale between output[{ss}] and input is {zoom_scale[axis - 2]}. "
                            f"Corresponding output roi range is ({zoomed_start}, {zoomed_end}).\n"
                            f"Please change overlap ({overlap}) or roi_size ({roi_size[axis-2]}) for axis-{axis-2}. "
                            "Tips: if overlap*roi_size*zoom_scale is an integer, it usually works."
                        )
                    original_idx_zoom[axis] = slice(
                        int(zoomed_start), int(zoomed_end), None)
                importance_map_zoom = resizer(importance_map.unsqueeze(0))[
                    0].to(compute_dtype)
                # store results and weights
                output_image_list[ss][original_idx_zoom] += importance_map_zoom * \
                    seg_prob[idx - slice_g]
                count_map_list[ss][original_idx_zoom] += (
                    importance_map_zoom.unsqueeze(0).unsqueeze(0).expand(
                        count_map_list[ss][original_idx_zoom].shape)
                )

    # account for any overlapping sections
    for ss in range(len(output_image_list)):
        output_image_list[ss] = (
            output_image_list[ss] / count_map_list.pop(0)).to(compute_dtype)

    # remove padding if image_size smaller than roi_size
    for ss, output_i in enumerate(output_image_list):
        if torch.isnan(output_i).any() or torch.isinf(output_i).any():
            warnings.warn(
                "Sliding window inference results contain NaN or Inf.")

        zoom_scale = [
            seg_prob_map_shape_d / roi_size_d for seg_prob_map_shape_d, roi_size_d in zip(output_i.shape[2:], roi_size)
        ]

        final_slicing: List[slice] = []
        for sp in range(num_spatial_dims):
            slice_dim = slice(
                pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
            slice_dim = slice(
                int(round(slice_dim.start *
                    zoom_scale[num_spatial_dims - sp - 1])),
                int(round(slice_dim.stop *
                    zoom_scale[num_spatial_dims - sp - 1])),
            )
            final_slicing.insert(0, slice_dim)
        while len(final_slicing) < len(output_i.shape):
            final_slicing.insert(0, slice(None))
        output_image_list[ss] = output_i[final_slicing]

    if dict_key is not None:  # if output of predictor is a dict
        final_output = dict(zip(dict_key, output_image_list))
    else:
        final_output = tuple(output_image_list)  # type: ignore
    final_output = final_output[0] if is_tensor_output else final_output

    if isinstance(inputs, MetaTensor):
        final_output = convert_to_dst_type(final_output, inputs, device=device)[
            0]  # type: ignore
    return final_output


class MySlidingInferer(monai.inferers.SlidingWindowInferer):
    def __init__(
        self,
        roi_size: Union[Sequence[int], int],
        input_size: Union[Sequence[int], int, None] = None,
        sw_batch_size: int = 1,
        overlap: Union[Sequence[float], float] = 0.25,
        mode: Union[BlendMode, str] = BlendMode.CONSTANT,
        sigma_scale: Union[Sequence[float], float] = 0.125,
        padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
        cval: float = 0.0,
        sw_device: Union[torch.device, str, None] = None,
        device: Union[torch.device, str, None] = None,
        progress: bool = False,
        cache_roi_weight_map: bool = False,
        cpu_thresh: Union[int, None] = None,
    ) -> None:
        super(MySlidingInferer, self).__init__(
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            mode=mode,
            sigma_scale=sigma_scale,
            padding_mode=padding_mode,
            cval=cval,
            sw_device=sw_device,
            device=device,
            progress=progress,
            cache_roi_weight_map=cache_roi_weight_map,
            cpu_thresh=cpu_thresh,
        )
        if input_size is None:
            input_size = roi_size
        self.input_size = input_size

    def __call__(
        self,
        inputs: torch.Tensor,
        network: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor], Dict[Any, torch.Tensor]]],
        *args: Any,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[Any, torch.Tensor]]:
        """
        Args:
            inputs: model input data for inference.
            network: target model to execute inference.
                supports callables such as ``lambda x: my_torch_model(x, additional_config)``
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.

        """

        device = kwargs.pop("device", self.device)
        if device is None and self.cpu_thresh is not None and inputs.shape[2:].numel() > self.cpu_thresh:
            # stitch in cpu memory if image is too large
            device = "cpu"

        return sliding_window_inference(
            inputs,
            self.roi_size,
            self.input_size,
            self.sw_batch_size,
            network,
            self.overlap,
            self.mode,
            self.sigma_scale,
            self.padding_mode,
            self.cval,
            self.sw_device,
            device,
            self.progress,
            self.roi_weight_map,
            *args,
            **kwargs,
        )
