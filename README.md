<p align="center">
<img src="https://raw.githubusercontent.com/prajjwal1/fluence/master/docs/logo.png" width="500">
<br />
<br />
<a href="https://github.com/prajjwal1/fluence/releases"><img alt="Latest Release" src="https://img.shields.io/pypi/v/fluence"/></a>
<a href="https://github.com/prajjwal1/fluence/blob/master/LICENSE"><img alt="Apache" src="https://img.shields.io/github/license/prajjwal1/fluence" /></a>
<a href="https://codecov.io/gh/prajjwal1/fluence"><img src="https://codecov.io/gh/prajjwal1/fluence/branch/master/graph/badge.svg" /></a>

Winner of Pytorch Global Hackathon 2020.

Fluence is a Pytorch based deep learning library focussed on providing computationally efficient, low resource methods and algorithms for NLP. Although the main focus is to provide support with transformers for NLP tasks, it can be extended with other domains and architectures as well. Currently in pre-alpha stage.

<details>
<summary>List of implemented papers</summary>

#### Adaptive Methods
- [Adaptive Attention Span in Transformers (ACL 2019)](https://arxiv.org/abs/1905.07799)
- [Adaptively Sparse Transformers (EMNLP 2019)](https://arxiv.org/abs/1909.00015)
- [Reducing Transformer Depth on Demand with Structured Dropout (ICLR 2020)](https://arxiv.org/abs/1909.11556)

#### Debiasing
- [Learning Robust Representations by Projecting Superficial Statistics Out (ICLR 2019)](https://openreview.net/pdf?id=rJEjjoR9K7)
-------------------------------------------------------------------------------

</details>

- [Installation](#installing)
- [Overview](#overview)

## Why Fluence ?
Fluence is targeted towards two main goals: 
1. **Compute efficiency**: Low resource research:
2. **Robustness**: Algorithms that either enhance our understanding of current methods or show where SoTA methods fail.

It is as straightforward to use as [HF Transformers](https://github.com/huggingface/transformers), and fully integrates with [Pytorch](https://github.com/pytorch/pytorch). Please note that the current modules (meta-trainer, siamese-trainer) which rely on inherited `Trainer` works with `transformers==3.0`. Newer version comes with a modified `Trainer`.

## Installing
For stable version:
```bash
pip3 install --user fluence
```

For development version (recommended):
```bash
git clone https://github.com/prajjwal1/fluence
cd fluence
python3 setup.py install --user
```

## Overview
The library contains implementation for the following approaches (many more to come):   
|  Module            |  Method with documentation
| -------------------------------------------------------------------------------------- | ----------------------------
| `fluence.adaptive` | [Adaptive Methods](https://github.com/prajjwal1/fluence/wiki/Adaptive-Methods)         |
| `fluence.datasets` | [Datasets](https://github.com/prajjwal1/fluence/wiki/datasets)                         |      
| `fluence.optim`    | [Optimizers](https://github.com/prajjwal1/fluence/wiki/Optimizers)                     |
| `fluence.sampling` | [Importance Sampling](https://github.com/prajjwal1/fluence/wiki/Importance-sampling)   |
| `fluence.models`   | [Siamese Methodology](https://github.com/prajjwal1/fluence/wiki/Siamese-Transformers), [Debiasing](https://github.com/prajjwal1/fluence/wiki/Debiasing)
| `fluence.prune` | [Pruning](https://github.com/prajjwal1/fluence/wiki/Pruning)|

## Documentation 
Please head to this [link](https://github.com/prajjwal1/fluence/wiki) to learn how you can integrate `fluence` with your workflow. Since it's an early release, there might be bugs. Please file an issue if you encounter one. Docs are a work-in-progress.

### Contribution
You can contribute by either filing an issue or sending a Pull Request (if you encounter any bug or want some features to be added). Please checkout the [contributing guide](https://github.com/prajjwal1/fluence/blob/master/CONTRIBUTING.md) for more details.


### Tests

Fluence comes with an [extensive test suite](https://github.com/prajjwal1/fluence/tree/master/tests) for high test coverage.
```
pytest tests/ -v
```

Author: Prajjwal Bhargava ([@prajjwal_1](https://twitter.com/prajjwal_1))
