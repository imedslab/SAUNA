import torch
import torch.nn as nn
import torch.nn.functional as F


def points_selection_hard(feat, prob, true, card=512, dis=100, **args):
    # point selection by ranking
    assert len(feat.shape) == 2, "feat should contains N*L two dims!"

    L = feat.shape[-1]
    # print(feat.shape, true.shape)
    feat = feat[true.view(-1, 1).repeat(1, L) > 0.5].view(-1, L)
    ############################################################
    # with torch.no_grad():
    prob = prob[true > 0.5].view(-1)
    idx = torch.sort(prob, dim=-1, descending=True)[1]
    # h = torch.index_select(feat, dim=0, index=idx[dis:dis+card])
    # l = torch.index_select(feat, dim=0, index=idx[-dis-card:-dis])
    ############################################################
    h = torch.index_select(feat, dim=0, index=idx[:card])
    l = torch.index_select(feat, dim=0, index=idx[-card:])
    return h, l


def points_selection_half(feat, prob, true, card=512, **args):
    # point selection by ranking
    assert len(feat.shape) == 2, "feat should contains N*L two dims!"

    L = feat.shape[-1]
    # print(feat.shape, true.shape)
    feat = feat[true.view(-1, 1).repeat(1, L) > 0.5].view(-1, L)

    ############################################################
    # with torch.no_grad():
    prob = prob[true > 0.5].view(-1)
    idx = torch.sort(prob, dim=-1, descending=False)[1]
    idx_l, idx_h = torch.chunk(idx, chunks=2, dim=0)

    sample1 = idx_h[torch.randperm(idx_h.shape[0])[:card]]
    sample2 = idx_l[torch.randperm(idx_l.shape[0])[:card]]
    ############################################################
    h = torch.index_select(feat, dim=0, index=sample1)
    l = torch.index_select(feat, dim=0, index=sample2)
    return h, l


class MLPSampler:
    def __init__(
        self, mode="hard",
        top=4, low=1,
        dis=0, num=512,
        select3=False,
        roma=False,
    ):
        self.top = top
        self.low = low
        self.dis = dis
        self.num = num
        self.roma = roma
        self.select = self.select3 if select3 else self.select2

        self.func = eval("points_selection_" + mode)

    @staticmethod
    def half(*args):
        return MLPSampler(mode="half", num=512).select(*args)

    def norm(self, *args, roma=False):
        args = [F.normalize(arg, dim=-1) for arg in args]
        if len(args) == 1:
            return args[0]
        return args

    def select(self, feat, pred, true, mask=None, kernel_size=5):
        assert feat.shape[-2:] == true.shape[-2:], "shape of feat & true donot match!"
        assert feat.shape[-2:] == pred.shape[-2:], "shape of feat & pred donot match!"
        # reshape embeddings
        feat = feat.clone().permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
        true = true.round()
        fh, fl = self.func(feat, pred, true, top=self.top,
                           low=self.low, dis=self.dis, card=self.num)
        return torch.cat([fh, fl], dim=0)

    def select2(self, feat, pred, true, mask=None, kernel_size=5):
        assert feat.shape[-2:] == true.shape[-2:], "shape of feat & true donot match!"
        assert feat.shape[-2:] == pred.shape[-2:], "shape of feat & pred donot match!"
        # reshape embeddings
        feat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
        true = true.round()
        back = (F.max_pool2d(true, (kernel_size, kernel_size),
                1, kernel_size//2) - true).round()

        fh, fl = self.func(feat, pred, true, top=self.top,
                           low=self.low, dis=self.dis, card=self.num)
        bh, bl = self.func(feat, 1 - pred, back, top=self.top,
                           low=self.low, dis=self.dis, card=self.num)
        return torch.cat([fh, fl, bh, bl], dim=0)

    def select3(self, feat, pred, true, mask=None, kernel_size=5):
        assert feat.shape[-2:] == true.shape[-2:], "shape of feat & true donot match!"
        assert feat.shape[-2:] == pred.shape[-2:], "shape of feat & pred donot match!"
        # reshape embeddings
        feat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
        true = true.round()
        dilate = F.max_pool2d(
            true, (kernel_size, kernel_size), 1, kernel_size//2).round()
        edge = (dilate - true).round()
        back = (1 - dilate).round() * mask.round()

        fh, fl = self.func(feat, pred, true, top=self.top,
                           low=self.low, dis=self.dis, card=self.num * 2)
        eh, el = self.func(feat, 1 - pred, edge, top=self.top,
                           low=self.low, dis=self.dis, card=self.num)
        bh, bl = self.func(feat, 1 - pred, back, top=self.top,
                           low=self.low, dis=self.dis, card=self.num)
        return torch.cat([fh, fl, eh, el, bh, bl], dim=0)
