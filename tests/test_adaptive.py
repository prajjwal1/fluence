import unittest

import torch
from torch import nn

from fluence.adaptive import AdaptiveSpan, EntmaxAlpha, Layerdrop

config = {
    "attn_span": 1024,
    "adapt_span_loss_coeff": 0.000005,
    "adapt_span_ramp": 32,
    "adapt_span_init": 0.002,
    "adapt_span_cache": True,
    "nb_heads": 12,
    "bs": 128,
    "mask_size": [20, 36],
}


class Test_Adaptive_Methods(unittest.TestCase):
    def test_adaptive_span(self):
        adaptive_span = AdaptiveSpan(**config)
        # Attention weights come from standard softmax
        attention_weights_1 = torch.randn(128, 12, 26, 36)
        attention_weights_2 = torch.randn(128, 12, 26, 20)

        # Feed the weights to apply the soft-masking function
        self.assertEqual(
            adaptive_span(attention_weights_1).shape, torch.Size([128, 12, 26, 36])
        )
        self.assertEqual(
            adaptive_span(attention_weights_2).shape, torch.Size([128, 12, 26, 20])
        )

        # Check the span characterstics
        self.assertTrue(adaptive_span.get_current_avg_span() > 0)
        self.assertEqual(adaptive_span.get_current_max_span(), 1024)

        # Clamp the parameter between range [0,1]
        adaptive_span.clamp_param()

        self.assertEqual(adaptive_span.get_trim_len(), 0)

    def test_layerdrop(self):
        net = nn.ModuleList([nn.Linear(2, 2) for i in range(3)])
        layerdrop = Layerdrop(net, 2)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        initial_params = list(layerdrop.module_list.parameters())
        loss = layerdrop(torch.rand(10, 2)).sum()
        self.assertTrue(loss is not None)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        updated_params = list(layerdrop.module_list.parameters())

    def test_entmax(self):
        num_attention_heads = 12
        entmax_alpha = EntmaxAlpha(num_attention_heads)
        attention_scores = torch.rand(128, 12, 26, 36)
        attention_scores = entmax_alpha(attention_scores)
        self.assertEqual(attention_scores.shape, torch.Size([128, 12, 26, 36]))


if __name__ == "__main__":
    unittest.main()
