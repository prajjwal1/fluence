import unittest

import torch

from fluence.pooling import MaxPooling, MeanPooling


class Test_Pooling(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.token_embeddings = torch.rand(8, 128, 768)
        self.attention_mask = torch.rand(8, 128)

    def test_mean_pooling(self):
        self.assertEqual(
            MeanPooling(self.token_embeddings, self.attention_mask).shape,
            torch.rand(8, 768).shape,
        )

    def test_max_pooling(self):
        self.assertEqual(
            MaxPooling(self.token_embeddings, self.attention_mask).shape,
            torch.rand(8, 768).shape,
        )


if __name__ == "__main__":
    unittest.main()
