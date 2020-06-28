import unittest

import torch
from transformers import GlueDataset, AutoTokenizer
from transformers import GlueDataTrainingArguments as DataTrainingArguments

from fluence.meta import MetaDataset


class Test_Meta(unittest.TestCase):
    def test_meta_dataset(self):
        MODEL_ID = "albert-base-v2"
        data_args = DataTrainingArguments(
            task_name="mrpc",
            data_dir="./tests/fixtures/tests_samples/MRPC",
            overwrite_cache=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer)
        meta_dataset = MetaDataset(train_dataset)
        self.assertEqual(len(meta_dataset[1000]), 2)
        self.assertEqual(meta_dataset[1000][0]["input_ids"].shape, torch.Size([128]))
        self.assertEqual(
            meta_dataset[1000][0]["attention_mask"].shape, torch.Size([128])
        )
        self.assertEqual(meta_dataset[1000][0]["labels"].item(), 0)
        self.assertEqual(meta_dataset[1000][1]["labels"].item(), 1)


if __name__ == "__main__":
    unittest.main()
