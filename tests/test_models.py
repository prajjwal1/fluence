import unittest

from types import SimpleNamespace
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import TrainingArguments, default_data_collator

from fixtures.models import CBOW
from fluence.models import OrthogonalTransformer, SiameseTransformer


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
        self.dataset = GlueDataset(self.data_args, self.tokenizer, mode="dev")
        self.config = AutoConfig.from_pretrained(
            self.MODEL_ID, num_labels=3, finetuning_task="mrpc")
        self.dataloader = DataLoader(self.dataset, batch_size=2, collate_fn=default_data_collator)

    def test_hex(self):
        model_a = AutoModel.from_pretrained(self.MODEL_ID)
        model_b = CBOW(self.config)
        self.config.batch_size = 2
        model = OrthogonalTransformer(model_a, model_b, self.config)
        batch = next(iter(self.dataloader))
        output = model(**batch)[1]
        self.assertEqual(output.shape, torch.Size([2, 3]))

    def test_siamese(self):
        model_args = {
            'model_name':self.MODEL_ID,
            'config_name':self.MODEL_ID,
            'tokenizer_name':self.MODEL_ID,
            'cache_dir': None
        }
        model_args = SimpleNamespace(**model_args)
        config = AutoConfig.from_pretrained(
            self.MODEL_ID, num_labels=3, finetuning_task="mrpc"
        )
        model = SiameseTransformer(model_args, config)
        batch = next(iter(self.dataloader))
        output = model(batch, batch)[1]
        self.assertEqual(output.shape, torch.Size([2, 3]))
