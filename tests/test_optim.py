import unittest

import torch
import torchvision

from fluence.optim import Lamb, Lookahead


class Test_Optim(unittest.TestCase):
    def test_lookahead(self):
        model = torchvision.models.AlexNet()
        base_optim = torch.optim.Adam(model.parameters())
        optim = Lookahead(base_optim, k=5, alpha=0.8)
        output = model(torch.rand(128, 3, 64, 64))
        optim.step()

    def test_lamb(self):
        model = torchvision.models.AlexNet()
        optim = Lamb(model.parameters())
        output = model(torch.rand(128, 3, 64, 64))
        optim.step()


if __name__ == "__main__":
    unittest.main()
