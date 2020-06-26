__all__ = ["Layerdrop", "Layerdrop_Cross"]

import torch
from torch import nn


class Layerdrop(nn.Module):
    """
    Implements [Reducing Transformer Depth on Demand with Structured Dropout]
    (https://arxiv.org/abs/1909.11556)

    Arguments:
        module_list (nn.ModuleList): List from which layers are to dropped.
        layers_to_drop (int): number of layers to drop

    Returns:
        feats: pruned features
    """

    def __init__(self, module_list, layers_to_drop):
        super(Layerdrop, self).__init__()
        self.module_list = module_list
        self.layers_to_drop = layers_to_drop
        self.length = len(module_list)

    def forward(self, feats, mask=None):
        x = torch.randint(0, self.length, (self.layers_to_drop,))
        for index, layer in enumerate(self.module_list):
            if index not in x:
                if not mask:
                    feats = layer(feats)
                else:
                    feats = layer(feats, mask)
        return feats


class Layerdrop_Cross(nn.Module):
    """
    This method is useful when layerdrop has to be
    used in multi modal settings (visual and linguistic)
    features

    Returns:
        lang_feats, visn_feats: pruned features for language and vision modality
    """

    def __init__(self, module_list, layers_to_drop=2):
        super(Layerdrop_Cross, self).__init__()
        self.module_list = module_list
        self.layers_to_drop = layers_to_drop
        self.length = len(module_list)

    def forward(self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask):
        x = torch.randint(0, self.length, (self.layers_to_drop,))
        for index, layer in enumerate(self.module_list):
            if index not in x:
                lang_feats, visn_feats = layer(
                    lang_feats, lang_attention_mask, visn_feats, visn_attention_mask
                )  #
        return lang_feats, visn_feats
