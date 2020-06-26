# Fluence
> Fluence is a Pytorch based deep learning library focussed on providing computationally efficient, low resource methods and algorithms. Although the main focus is to provide support with transformers, it can be extended with other architectures as well.


![badge](https://github.com/prajjwal1/fluence/workflows/build/badge.svg)
[![PyPI version](https://badge.fury.io/py/fluence.svg)](https://badge.fury.io/py/fluence)

# Installing

`pip install fluence`

The library contains implementation for the following approaches (many more to come):
- Adaptive Methods
    - [Adaptive Attention Span in Transformers](https://arxiv.org/abs/1905.07799)
    - [Adaptively Sparse Transformers](https://arxiv.org/abs/1909.00015)
    - [Reducing Transformer Depth on Demand with Structured Dropout](https://arxiv.org/abs/1909.11556)

- Optimizers: 
    - Lamb
    - Lookahead
    
- Importance Sampling:
    - Clustering

# Documentation 
Please head to this [link](prajjwal1.github.io/fluence) to learn how you can integrate fluence with your workflow. Since it's an early release, there might be bugs here and there. Please file an issue if you encounter one.

## Minimal Examples
Fluence is easy to use and fully compatible with Huggingface transformers. Here are some of the examples


### Using Adaptive Attention Span
```
import torch
from fluence.adaptive.adaptive_span import AdaptiveSpan
config = {'attn_span': 1024, 'adapt_span_loss_coeff': 0.000005, 'adapt_span_ramp': 32,
                      'adapt_span_init': 0.002, 'adapt_span_cache': True, 'nb_heads': 12,'bs': 128,
                      'mask_size': [20,36]}
adaptive_span = AdaptiveSpan(**config)

attention_scores_0 = torch.randn(128,12,26,36) # These scores come from softmax
attention_scores_1 = torch.randn(128,12,26,20) # (Optional) These scores come from softmax
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

### fluence.sampling

```bash
from fluence.sampling.clustering import Clustering_Arguments, Clustering_Processor
# Similar to Huggingface Training Arguments
clustering_args = Clustering_Arguments(
        batch_size=32,
        num_clusters_elements=32,
        embedding_path="/home/nlp/experiments/cls_embeddings_mnli.pth",
        num_clusters=8,
        cluster_output_path="/home/nlp/experiments/tmp/c.pth",
    )

clustering_proc = Clustering_Processor(vars(clustering_obj))

# Now perform sampling by data percentage,  or centroids
cluster_indices = clustering_proc.get_cluster_indices_by_pct(
        clustering_args.data_pct, embeddings.shape[0]
    )
# By number of clusters
cluster_indices = clustering_proc.get_cluster_indices_by_num(
    clustering_args.num_clusters_elements
)
# Or centroids
cluster_indices = clustering_proc.get_cluster_indices_by_num(
        clustering_args.num_clusters_elements
    )

train_dataset = GlueDataset(data_args, tokenizer)
train_dataset = torch.utils.data.Subset(train_dataset, cluster_indices)
```

Author: Prajjwal Bhargava ([@prajjwal_1](https://twitter.com/prajjwal_1))
