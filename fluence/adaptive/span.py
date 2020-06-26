__all__ = ["AdaptiveSpan"]

import math

import torch
import torch.nn as nn


class AdaptiveSpan(nn.Module):
    """
    Implements `Adaptive Attention Span in Transformers`
                [Paper](https://arxiv.org/abs/1905.07799)

    Arguments:
        attn_span (int): specifies the maximum attention span
        adapt_span_loss_coeff (float): regulates the initial value of
                                        adapt_span_loss
        adapt_span_ramp (int): offset value
        adapt_span_init (float): initial additive value for the
                                 main parameter
        adapt_span_cache (bool): determines working of caching
        nb_heads (int): number of attention heads
        bs (int): batch size
        mask_size (list): a list containing last dimension of possible
                        attention scores

    Example::

        >>> config = {'attn_span': 1024,
                     'adapt_span_loss_coeff': 0.000005, 'adapt_span_ramp': 32,
                     'adapt_span_init': 0.002, 'adapt_span_cache': True,
                     'nb_heads': 12,'bs': 128, 'mask_size': [20,36]}
        >>> adaptive_span = AdaptiveSpan(**config)
        >>> adaptive_span(torch.randn(128,12,26,36)).shape
        >>> adaptive_span(torch.randn(128,12,26,20)).shape
        >>> adaptive_span.get_current_avg_span()
        >>> adaptive_span.get_current_max_span()
        >>> adaptive_span.get_trim_len()
        >>> adaptive_span.clamp_param()
    """

    def __init__(
        self,
        attn_span,
        adapt_span_loss_coeff,
        adapt_span_ramp,
        adapt_span_init,
        adapt_span_cache,
        nb_heads,
        bs,
        mask_size,
    ):

        super(AdaptiveSpan, self).__init__()
        self.attn_span = attn_span
        self.ramp_size = adapt_span_ramp
        self.bs = bs
        self.nb_heads = nb_heads
        self.init_val = adapt_span_init
        self.adapt_cache = adapt_span_cache
        self.loss_coeff = adapt_span_loss_coeff
        self.shape = (self.bs, self.nb_heads, 1, 1)

        self.current_val = nn.Parameter(
            torch.nn.init.kaiming_normal_(torch.empty(*self.shape)) + self.init_val
        )  # [bs,nb_heads,1,1]
        self.mask_size = mask_size

        mask_template_0 = torch.linspace(
            1 - self.mask_size[0], 0, steps=self.mask_size[0]
        )  # [attn_span]
        self.register_buffer("mask_template_0", mask_template_0)

        if len(self.mask_size) > 1:
            mask_template_1 = torch.linspace(
                1 - self.mask_size[1], 0, steps=self.mask_size[1]
            )
            self.register_buffer("mask_template_1", mask_template_1)

    def mask_forward(self, x):
        """
        Computes the mask and performs the multiplication operation
        with attention weights
        """
        mask_size = x.size(3)
        if mask_size == self.mask_size[0]:
            mask = self.mask_template_0 + self.current_val * mask_size
        else:
            mask = self.mask_template_1 + self.current_val * mask_size
        mask = mask / self.ramp_size + 1
        mask = mask.clamp(0, 1)
        if x.size(0) == mask.size(0):
            x = x * mask  # [bs, nb_heads, 36, 64]) [bs, nb_heads, 1, 64]
            return x
        else:
            return x

    def get_current_avg_span(self, include_ramp=True):
        """
        Outputs average span
        """
        current_size = math.ceil(self.current_val.mean().item() * self.attn_span)
        if include_ramp:
            current_size += self.ramp_size
        current_size = max(0, min(self.attn_span, current_size))
        return current_size

    def get_current_max_span(self, include_ramp=True):
        """
        Determines maximum span
        """
        current_size = math.ceil(self.current_val.max().item() * self.attn_span)
        if include_ramp:
            current_size += self.ramp_size
        current_size = max(0, min(self.attn_span, current_size))
        return current_size

    def clamp_param(self):
        """
        Clamps the values of parameter to stay between 0 and 1
        """
        self.current_val.data.clamp_(0, 1)

    def get_trim_len(self):
        """
        Outputs length to be trimmed
        """
        L = self.attn_span
        trim_len = min(L - 1, L - self.get_current_max_span())
        trim_len = math.floor(trim_len / 64) * 64
        return trim_len

    def get_cache_size(self):
        """
        Determine how long the cache should be
        """
        if self.adapt_cache:
            trim_len = self.get_trim_len()
            return min(self.attn_span, self.attn_span - trim_len + 64)
        else:
            return self.attn_span

    def get_loss(self):
        """
        A loss term for regularizing the span length
        """
        return self.loss_coeff * self.attn_span * self.current_val.mean()

    def forward(self, attn):
        attn = self.mask_forward(attn)
        attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)
        return attn
