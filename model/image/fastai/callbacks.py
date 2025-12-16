from __future__ import annotations

from fastai.callback.core import Callback
import torch


class GradientClipping(Callback):
    def __init__(self, max_norm: float = 0.5, norm_type: float = 2.0):
        self.max_norm, self.norm_type = max_norm, norm_type

    def after_backward(self):
        torch.nn.utils.clip_grad_norm_(self.learn.model.parameters(), self.max_norm, self.norm_type)


class NoiseAnnealCallback(Callback):
    """
    Disable any transform instance of a given class in the last pct_stop of training.
    Designed for custom train-only noise transforms.
    """

    def __init__(self, pct_stop: float = 0.8, transform_cls=object):
        self.pct_stop = pct_stop
        self.transform_cls = transform_cls

    def before_fit(self):
        iters_per_epoch = len(self.dls.train)
        total_iters = self.n_epoch * iters_per_epoch
        self.cutoff_iter = int(total_iters * self.pct_stop)
        # keep original Pipeline and a filtered copy without the noise transform
        self.orig_after_item = self.dls.train.after_item
        filtered = [t for t in self.orig_after_item if not isinstance(t, self.transform_cls)]
        self.filtered_after_item = type(self.orig_after_item)(filtered)

    def before_batch(self):
        if self.learn.iter >= self.cutoff_iter:
            self.dls.train.after_item = self.filtered_after_item
        else:
            self.dls.train.after_item = self.orig_after_item
