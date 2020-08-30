import unittest

import torch
from torch import nn

from fluence.optim import Lamb, Lookahead

class Model(nn.Module):
    def __init__(self):
        self.model = nn.Sequential(nn.Linear(8, 8),
                                   nn.ReLU(),
                                   nn.Linear(8, 8))

    def forward(self, x):
        return self.model(x)

class Test_Optim(unittest.TestCase):
    def test_lookahead(self):
        model = Model()
        base_optim = torch.optim.Adam(model.parameters())
        optim = Lookahead(base_optim, k=5, alpha=0.8)
        output = model(torch.rand(128, 3, 64, 64))
        optim.step()

    def test_lamb(self):
        model = Model()
        optim = Lamb(model.parameters())
        output = model(torch.rand(128, 3, 64, 64))
        optim.step()


if __name__ == "__main__":
    unittest.main()
