# coding=utf-8
# Author: Prajjwal Bhargava

import logging
from collections import defaultdict

import torch
from tqdm import trange

logger = logging.getLogger(__name__)


class MetaDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.args = self.dataset.args
        self.processor = self.dataset.processor
        self.features = self.dataset.features
        self.label_list = self.dataset.label_list
        self.output_mode = self.dataset.output_mode
        self.indices_mapping = self._get_indices_mapping()
        self.num_labels = len(self.indices_mapping.keys())
        self.min_len = self.get_len()
        self.data = self.get_tensorized_data()

    def get_len(self):
        min_len = float("inf")
        for values in self.indices_mapping.values():
            min_len = min(len(values), min_len)
        return min_len

    def __len__(self):
        return self.min_len

    def _get_indices_mapping(self):
        indices_mapping = {}
        for idx, data in enumerate(self.dataset):
            indices_mapping.update({idx: data.label})

        temp_mapping = defaultdict(list)
        for key, value in sorted(indices_mapping.items()):
            temp_mapping[value].append(key)

        indices_mapping = temp_mapping
        del temp_mapping
        return indices_mapping

    def __getitem__(self, idx):
        return self.data[idx]

    def get_tensorized_data(self):
        tensorized_data = []
        dtype = torch.long
        logging.info("**** Preparing Meta Dataset ****")
        for idx in trange(self.min_len):
            res = []
            for label in range(self.num_labels):
                data = self.features[self.indices_mapping[label][idx]]
                res.append(
                    {
                        "input_ids": torch.tensor(data.input_ids, dtype=dtype),
                        "attention_mask": torch.tensor(
                            data.attention_mask, dtype=dtype
                        ),
                        "token_type_ids": torch.tensor(
                            data.token_type_ids, dtype=dtype
                        ),
                        "labels": torch.tensor(data.label, dtype=dtype),
                    }
                )
            tensorized_data.append(res)
        return tensorized_data
