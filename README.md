# Fluence
> Fluence is a deep learning library based on Pytorch for attention based approaches. It's a toolkit which I use for my own research.


![](https://github.com/prajjwal1/fluence/workflows/CI/badge.svg)
[![PyPI version](https://badge.fury.io/py/fluence.svg)](https://badge.fury.io/py/fluence)

# Installing

`pip install fluence`

The library contains implementation for the following approaches (many more to come):
- [Adaptive Attention Span in Transformers](https://arxiv.org/abs/1905.07799)
- [Adaptively Sparse Transformers](https://arxiv.org/abs/1909.00015)
- [Reducing Transformer Depth on Demand with Structured Dropout](https://arxiv.org/abs/1909.11556)
- Optimizers: Lamb, Lookahead (Pytorch doesn't have it)

# Code Structure
```
fluence
    - adaptive     # Implements Adaptive Modules
    - models       # Models
    - optimizers   # optimizers 
```

# Documentation 
Please head to this [link](prajjwal1.github.io/fluence) to learn how you can integrate fluence with your workflow. Since it's an early release, there might be bugs here and there. Please file an issue if you encounter one.

## Usage
Right now, it consists of major adaptive computation approaches which have been tested with transformers. Fluence is easy to use. Here are some of the examples


### Using Adaptive Attention Span
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

### Using Entmax as a replacement for softmax with learnable alpha values

```
from fluence.adaptive.entmax import *
num_attention_heads = 12
entmax_alpha = EntmaxAlpha(num_attention_heads)
attention_scores = entmax_alpha(att_scores=torch.rand(128,12,26,36)) 
```

### Using Layerdrop

```
from fluence.adaptive.layerdrop import Layerdrop
from torch import nn
net = nn.ModuleList([nn.Linear(2, 2) for i in range(3)])
layers_to_drop = 2
layerdrop = Layerdrop(net, layers_to_drop)
output = layerdrop(torch.rand(10,2))
```

### fluence.optim
```
from fluence.optim.lamb import Lamb
from fluence.optim.lookahead import Lookahead

model = torchvision.models.AlexNet()                        # Can be a transformer
base_optim = Lamb(params=model.parameters(),lr=1e-5, weight_decay=1.2e-6, min_trust=0.25)
optim = Lookahead(base_optimizer=base_optim, k=5, alpha=0.8)
```

#### Acknowledgements
- [Hugging face Transformer](https://github.com/huggingface/transformers/)
- [Adaptive Attention Span for Transformers](https://github.com/facebookresearch/adaptive-span)
- [entmax](https://github.com/deep-spin/entmax)
- [LXMERT](https://github.com/airsplay/lxmert)

Author: Prajjwal Bhargava ([@prajjwal_1](https://twitter.com/prajjwal_1))
