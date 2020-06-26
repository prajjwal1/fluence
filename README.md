# Fluence
> Fluence is a Pytorch based deep learning library focussed on providing computationally efficient, low resource methods and algorithms. Although the main focus is to provide support with transformers, it can be extended with other architectures as well.


![badge](https://github.com/prajjwal1/fluence/workflows/build/badge.svg)
[![PyPI version](https://badge.fury.io/py/fluence.svg)](https://badge.fury.io/py/fluence)

# Installing

`python3 setup.py install --user`

The library contains implementation for the following approaches (many more to come):
- [Adaptive Methods](https://github.com/prajjwal1/fluence/wiki/Importance-sampling)
    - [Adaptive Attention Span in Transformers](https://arxiv.org/abs/1905.07799)
    - [Adaptively Sparse Transformers](https://arxiv.org/abs/1909.00015)
    - [Reducing Transformer Depth on Demand with Structured Dropout](https://arxiv.org/abs/1909.11556)

- [Meta Learning](https://github.com/prajjwal1/fluence/wiki/Meta-Learning)

- [Optimizers](https://github.com/prajjwal1/fluence/wiki/Optimizers): 
    - [Lamb](https://arxiv.org/abs/1904.00962)
    - [Lookahead](https://arxiv.org/abs/1907.08610)
    
- [Importance Sampling](https://github.com/prajjwal1/fluence/wiki/Importance-sampling):
    - Clustering


# Documentation 
Please head to this [link](https://github.com/prajjwal1/fluence/wiki) to learn how you can integrate fluence with your workflow. Since it's an early release, there might be bugs here and there. Please file an issue if you encounter one.

# Tests
Tests can be run with pytest
```
pytest tests/
```

Author: Prajjwal Bhargava ([@prajjwal_1](https://twitter.com/prajjwal_1))
