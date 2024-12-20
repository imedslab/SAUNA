from typing import List, Set, Tuple, Iterable, cast
from functools import partial
from operator import itemgetter

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt as eucl_distance
from torchvision import transforms


def simplex(t: torch.Tensor, axis=1) -> bool:
    _sum = cast(torch.Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def uniq(a: torch.Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: torch.Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def one_hot(t: torch.Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


def probs2class(probs: torch.Tensor) -> torch.Tensor:
    b, _, *img_shape = probs.shape
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, *img_shape)
    return res


def class2one_hot(seg: torch.Tensor, K: int) -> torch.Tensor:
    assert sset(seg, list(range(K))), (uniq(seg), K)

    if len(seg.shape) == 4:
        seg = torch.squeeze(seg, dim=1)
    b, *img_shape = seg.shape
    # type: Tuple[int, ...]

    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32,
                      device=device).scatter_(1, seg[:, None, ...].long(), 1)

    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)
    return res


def probs2one_hot(probs: torch.Tensor) -> torch.Tensor:
    _, K, *_ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), K)
    assert res.shape == probs.shape
    assert one_hot(res)
    return res


def one_hot2dist(
    seg: np.ndarray,
    resolution: Tuple[float, float, float] = None,
    dtype=None,
) -> np.ndarray:
    assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(bool)

        if posmask.any():
            negmask = ~posmask
            res[k] = eucl_distance(negmask, sampling=resolution) * negmask \
                - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res


def gt_transform(resolution: Tuple[float, ...], K: int):
    return transforms.Compose([
        lambda img: np.array(img)[...],
        lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],
        # Add one dimension to simulate batch
        partial(class2one_hot, K=K),
        itemgetter(0),
        # Then pop the element to go back to img shape
    ])


def dist_map_transform(resolution: Tuple[float, ...], K: int):
    return transforms.Compose([
        gt_transform(resolution, K),
        lambda t: t.cpu().numpy(),
        partial(one_hot2dist, resolution=resolution),
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])


class BoundaryLoss():
    def __init__(self, **kwargs):
        # self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: torch.Tensor, dist_maps: torch.Tensor) -> torch.Tensor:
        probs_dim_one = (
            probs.shape[1] == 1 and dist_maps.shape[1] == 2 and len(self.idc) == 1)
        assert simplex(probs) or probs_dim_one
        assert not one_hot(dist_maps)

        p_index = [0] if probs_dim_one else self.idc
        pc = probs[:, p_index, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = torch.einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()
        return loss
