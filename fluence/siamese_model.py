import logging

import torch
from torch import nn
from transformers import AutoModel, AutoModelForSequenceClassification, PreTrainedModel

logger = logging.getLogger(__name__)


class PredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((8, 128))
        self.dense = nn.Linear(4096, len(config.id2label))

    def forward(self, features):
        features = self.pool(features)
        features = features.view(features.shape[0] // 4, -1)
        features = self.dense(features)
        return features


class SiameseTransformer(nn.Module):
    def __init__(self, args, config):
        super(SiameseTransformer, self).__init__()
        self.args = args
        self.model_a = AutoModelForSequenceClassification.from_pretrained(
            self.args.model_name, config=config, cache_dir=self.args.cache_dir
        )
        self.model_b = AutoModelForSequenceClassification.from_pretrained(
            self.args.model_name, config=config, cache_dir=self.args.cache_dir
        )

        self.loss_fct = nn.CrossEntropyLoss()
        # self.cls = PredictionHeadTransform(config)
        # self.cls = nn.Linear(len(config.id2label), len(config.id2label))
        # if self.args.freeze_a:
        #    logger.info("**** Freezing Model A ****")
        #    for param in self.model_a.encoder.parameters():
        #        param.requires_grad = False

        # if self.args.freeze_b:
        #    logger.info("**** Freezing Model B ****")
        #    for param in self.model_b.encoder.parameters():
        #        param.requires_grad = False

    def forward(self, a, b):
        # labels = input_a['labels']
        # input_a.pop('labels')
        # input_b.pop('labels')
        output_a = self.model_a(**a)  # [bs, seq_len, 768]
        output_b = self.model_b(**b)
        outputs = []
        for i in range(len(output_a)):
            outputs.append(output_a[i] + output_b[i])

        # concat_output = torch.cat([output_a[1], output_b[1]])
        # logits = self.cls(concat_output)
        # outputs.append(logits)
        # loss = self.loss_fct(logits, labels)
        return outputs
