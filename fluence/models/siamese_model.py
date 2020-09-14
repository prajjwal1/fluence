import logging

import torch
from torch import nn
from transformers import AutoModel, AutoModelForSequenceClassification, PreTrainedModel

logger = logging.getLogger(__name__)


class SiameseTransformer(nn.Module):
    def __init__(self, args, config):
        super(SiameseTransformer, self).__init__()
        self.args = args
        self.model_a = AutoModel.from_pretrained(
            self.args.model_name, config=config, cache_dir=self.args.cache_dir
        )
        self.model_b = AutoModel.from_pretrained(
            self.args.model_name, config=config, cache_dir=self.args.cache_dir
        )
        self.linear = nn.Linear(config.hidden_size * 3, config.num_labels)

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, a, b):
        labels = a["labels"]
        a.pop("labels")
        b.pop("labels")
        output_a = self.model_a(**a)[1]  # [bs, seq_len, 768]
        output_b = self.model_b(**b)[1]
        output = torch.cat([output_a, output_b, output_a - output_b], dim=1)
        logits = self.linear(output)
        loss = self.loss_fct(logits, labels)
        return loss, logits


class SiameseTransformerAdd(nn.Module):
    def __init__(self, args, config):
        super(SiameseTransformerAdd, self).__init__()
        self.args = args
        self.model_a = AutoModelForSequenceClassification.from_pretrained(
            self.args.model_name, config=config, cache_dir=self.args.cache_dir
        )
        self.model_b = AutoModelForSequenceClassification.from_pretrained(
            self.args.model_name, config=config, cache_dir=self.args.cache_dir
        )

    def forward(self, a, b):
        output_a = self.model_a(**a)
        output_b = self.model_b(**b)
        outputs = []
        for i in range(len(output_a)):
            outputs.append(output_a[i] + output_b[i])

        return outputs
