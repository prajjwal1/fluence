import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

class Pruner():
    def __init__(self, model, layer_type, pct):
        self.model = model
        self.layer_type = layer_type
        self.pct = pct
        self._method_type = {'random': prune.RandomUnstructured,
                       'l1': prune.L1Unstructured,
                       'ln': prune.LnStructured
                      }
        self._parameters_to_prune = []
        self._parameters_to_prune_names = []

        _lin_cnt, _conv_cnt = 0, 0
        for name, module in model.named_modules():
            if 'linear' in layer_type:
                if isinstance(module, nn.Linear):
                    self._parameters_to_prune.append((module, 'weight'))
                    self._parameters_to_prune_names.append(name)
                    _lin_cnt += 1
            if 'conv' in layer_type:
                if isinstance(module, nn.Conv2d):
                    self._parameters_to_prune.append((module, 'weight'))
                    self._parameters_to_prune_names.append(name)
                    _conv_cnt += 1
        print("Detected {} Linear layers".format(_lin_cnt))
        print("Detected {} Conv layers".format(_conv_cnt))

    def perform_pruning(self, method, **kwargs):
        chosen_method = self._method_type[method]
        prune.global_unstructured(
            self._parameters_to_prune,
            pruning_method=chosen_method,
            amount=self.pct,
        )

    def make_permanent(self):
        for module in self._parameters_to_prune:
            prune.remove(module[0], 'weight')
