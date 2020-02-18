# Fluence
> Fluence is a deep learning library based on Pytorch for attention based approaches.


## Installing

`pip install fluence`

The library contains implementation for the following approaches (many more to come):
- [Adaptive Attention Span in Transformers](https://arxiv.org/abs/1905.07799)
- [Adaptively Sparse Transformers](https://arxiv.org/abs/1909.00015)
- [Reducing Transformer Depth on Demand with Structured Dropout](https://arxiv.org/abs/1909.11556)

## Documentation 
Please head to this [link](prajjwal1.github.io/fluence) to learn how you can integrate fluence with your workflow

## Usage
Right now, it consists of major adaptive computation approaches which have been tested with transformers. Fluence is easy to use. Here are some of the examples


#### Using Adaptive Attention Span
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

#### Using Entmax as a replacement for softmax with learnable alpha values

```
from fluence.adaptive.entmax import *
num_attention_heads = 12
entmax_alpha = EntmaxAlpha(num_attention_heads)
attention_scores = entmax_alpha(att_scores=torch.rand(128,12,26,36)) 
```

#### Using Layerdrop

```
from fluence.adaptive.layerdrop import LayerDrop
from torch import nn
net = nn.ModuleList([nn.Linear(2, 2) for i in range(3)])
layers_to_drop = 2
layerdrop = LayerDrop(net, layers_to_drop)
output = layerdrop(torch.rand(10,2))
```

Author: Prajjwal Bhargava ([@prajjwal_1](https://twitter.com/prajjwal_1))
