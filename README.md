# Title
> summary


## Fluence
Fluence is a deep learning library based on Pytorch for adaptive computation approaches.

## Install
`pip install fluence`

The library contains implementation for the following approaches (many more to come):
- [Adaptive Attention Span in Transformers](https://arxiv.org/abs/1905.07799)
- [Adaptively Sparse Transformers](https://arxiv.org/abs/1909.00015)
- [Reducing Transformer Depth on Demand with Structured Dropout](https://arxiv.org/abs/1909.11556)

## How to use

### Using Adaptive Attention Spans in a transformer

```
import torch
from fluence.adaptive.adaptive_span import AdaptiveSpan
config = {'attn_span': 1024, 'adapt_span_loss_coeff': 0.000005, 'adapt_span_ramp': 32,
                      'adapt_span_init': 0.002, 'adapt_span_cache': True, 'nb_heads': 12,'bs': 128,
                      'mask_size': [20,36]}
adaptive_span = AdaptiveSpan(**config)
adaptive_span.get_current_avg_span() # Returns average span
adaptive_span.get_current_max_span() # Returns maximum span
adaptive_span.get_trim_len() # Returns length that can be trimmed
adaptive_span.clamp_param() # Clamps values of parameter to stay between [0,1]

attention_scores_0 = torch.randn(128,12,26,36) # These scores come from softmax
attention_scores_1 = torch.randn(128,12,26,20) # These scores come from softmax
adaptive_span(attention_scores_0).shape # Soft masking function is multiplied
adaptive_span(attention_scores_1).shape
```

Define the following in __init__ of BertAttention
1. Set it as an attribute
    ```
    if self.adapt_span_bool:
        self.adaptive_span = AdaptiveSpan(**config)
    ```
2. Use the adapt_span_loss with the current loss function
```
adapt_span_loss = 0.
for l in self.model.layer: # Should be a nn.ModuleList to iterate
        adapt_span_loss += l.attention.adaptive_span.get_loss() #attention is the BertAttention class
```
3. Perform clamping
```
for l in self.model.layer:
        l.attention.self.adaptive_span.clamp_param()
```
4. Get attention span
```
for layer_idx, i in enumerate(self.model.layer):
        l = i.attention.adaptive_span.get_current_avg_span()
```
