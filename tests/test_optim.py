import unittest

import torch
from torch import nn

from fluence.optim import Lamb, Lookahead

class Test_Optim(unittest.TestCase):
    def test_lookahead(self):
        model = nn.Linear(8, 8)
        base_optim = torch.optim.Adam(model.parameters())
        optim = Lookahead(base_optim, k=5, alpha=0.8)
        output = model(torch.rand(128, 3, 8, 8))
        optim.step()

    def test_lamb(self):
        model = nn.Linear(8, 8)
        optim = Lamb(model.parameters())
        output = model(torch.rand(128, 3, 8, 8))
        optim.step()


if __name__ == "__main__":
    unittest.main()
