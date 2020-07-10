import os
import unittest

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import default_data_collator

from fluence.meta import MetaArguments, MetaDataset, MetaTrainer


class Test_Meta(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "albert-base-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_ID)

    def test_meta_dataset(self):
        MODEL_ID = "albert-base-v2"
        data_args = DataTrainingArguments(
            task_name="mrpc",
            data_dir="./tests/fixtures/tests_samples/MRPC",
            overwrite_cache=True,
        )
        train_dataset = GlueDataset(data_args, tokenizer=self.tokenizer)
        meta_dataset = MetaDataset(train_dataset)
        self.assertEqual(len(meta_dataset[1000]), 2)
        self.assertEqual(meta_dataset[1000][0]["input_ids"].shape, torch.Size([128]))
        self.assertEqual(
            meta_dataset[1000][0]["attention_mask"].shape, torch.Size([128])
        )
        self.assertEqual(meta_dataset[1000][0]["labels"].item(), 0)
        self.assertEqual(meta_dataset[1000][1]["labels"].item(), 1)

    def test_meta_trainer(self):
        meta_args = MetaArguments(
            data_dir="./tests/fixtures/tests_samples/MRPC",
            output_dir="./examples",
            train_task="mrpc",
            eval_task="SST-2",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=512,
            overwrite_cache=True,
            no_cuda=True,
        )
        meta_args.task_name = meta_args.train_task
        eval_dataset = GlueDataset(meta_args, tokenizer=self.tokenizer, mode="dev")
        meta_trainer = MetaTrainer(
            model=self.model,
            args=meta_args,
            eval_dataset=eval_dataset,
            eval_data_collator=default_data_collator,
        )
        result = meta_trainer.evaluate()
        self.assertTrue(result["eval_loss"] > 0.5)


if __name__ == "__main__":
    unittest.main()
