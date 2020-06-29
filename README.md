# Fluence
> Fluence is a Pytorch based deep learning library focussed on providing computationally efficient, low resource methods and algorithms. Although the main focus is to provide support with transformers for NLP tasks, it can be extended with other domains and architectures as well. Currently in pre-alpha stage.


![badge](https://github.com/prajjwal1/fluence/workflows/build/badge.svg)
[![PyPI version](https://badge.fury.io/py/fluence.svg)](https://badge.fury.io/py/fluence)

# Installing
For stable (recommended) version:
```bash
pip3 install --user fluence
```

For development version:
```bash
git clone https://github.com/prajjwal1/fluence
cd fluence
python3 setup.py install --user
```

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


### Documentation 
Please head to this [link](https://github.com/prajjwal1/fluence/wiki) to learn how you can integrate fluence with your workflow. Since it's an early release, there might be bugs here and there. Please file an issue if you encounter one.

### Contribution
I'd really appreciate if you can file an issue or send a PR if you encounter any bug or want some features to be added. Please checkout the [contributing guide](https://github.com/prajjwal1/fluence/blob/master/CONTRIBUTING.md) for more details.


### Tests
Tests can be run with pytest
```
pytest tests/
```

Author: Prajjwal Bhargava ([@prajjwal_1](https://twitter.com/prajjwal_1))
