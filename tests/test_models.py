import unittest

import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import TrainingArguments, default_data_collator

from fixtures.models import CBOW
from fluence.models import OrthogonalTransformer


class Test_Model(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "albert-base-v2"
        self.data_args = DataTrainingArguments(
            task_name="mrpc",
            data_dir="./tests/fixtures/tests_samples/MRPC",
            overwrite_cache=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.train_dataset = GlueDataset(self.data_args, self.tokenizer)

    def test_hex(self):
        config = AutoConfig.from_pretrained(
            self.MODEL_ID, num_labels=3, finetuning_task="mrpc"
        )
        model_a = AutoModel.from_pretrained(self.MODEL_ID)
        model_b = CBOW(config)
        config.batch_size = 64
        model = OrthogonalTransformer(model_a, model_b, config)
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            collate_fn=default_data_collator,
        )
        batch = next(iter(dataloader))
        output = model(**batch)[1]
        self.assertEqual(output.shape, torch.Size([64, 3]))
