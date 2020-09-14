import os
import unittest

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import TrainingArguments

from fluence.datasets import SiameseGlueDataset, siamese_data_collator
from fluence.models import SiameseTransformer, SiameseTransformerAdd
from fluence.utils.siamese_utils import SiameseModelArguments, SiameseTrainer


class Test_Siamese(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "albert-base-v2"
        self.data_args = DataTrainingArguments(
            task_name="mrpc",
            data_dir="./tests/fixtures/tests_samples/MRPC",
            overwrite_cache=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.train_dataset = SiameseGlueDataset(self.data_args, self.tokenizer)
        self.eval_dataset = SiameseGlueDataset(
            self.data_args, self.tokenizer, mode="dev"
        )

    def test_dataset(self):
        self.assertEqual(len(self.train_dataset[0]["a"].input_ids), 128)
        self.assertEqual(len(self.train_dataset[0]["b"].input_ids), 128)
        self.assertEqual(
            self.train_dataset[100]["a"].label, self.train_dataset[100]["b"].label,
        )

    def test_siamese(self):
        model_args = SiameseModelArguments(
            model_name=self.MODEL_ID,
            config_name=self.MODEL_ID,
            tokenizer_name=self.MODEL_ID,
        )
        config = AutoConfig.from_pretrained(
            self.MODEL_ID, num_labels=3, finetuning_task="mrpc"
        )
        model = SiameseTransformer(model_args, config)
        training_args = TrainingArguments(output_dir="./tests", do_eval=True)
        trainer = SiameseTrainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=siamese_data_collator,
        )
        result = trainer.evaluate()
        self.assertTrue(result["eval_loss"] > 0.5)

    def test_siamese_add(self):
        model_args = SiameseModelArguments(
            model_name=self.MODEL_ID,
            config_name=self.MODEL_ID,
            tokenizer_name=self.MODEL_ID,
        )
        config = AutoConfig.from_pretrained(
            self.MODEL_ID, num_labels=3, finetuning_task="mrpc"
        )
        model = SiameseTransformerAdd(model_args, config)
        training_args = TrainingArguments(output_dir="./tests", do_eval=True)
        trainer = SiameseTrainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=siamese_data_collator,
        )
        result = trainer.evaluate()
        self.assertTrue(result["eval_loss"] > 0.5)

    def test_siamese_add(self):
        model_args = SiameseModelArguments(
            model_name=self.MODEL_ID,
            config_name=self.MODEL_ID,
            tokenizer_name=self.MODEL_ID,
        )
        config = AutoConfig.from_pretrained(
            self.MODEL_ID, num_labels=3, finetuning_task="mrpc"
        )
        model = SiameseTransformer(model_args, config)
        training_args = TrainingArguments(output_dir="./tests", do_eval=True)
        trainer = SiameseTrainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=siamese_data_collator,
        )
        result = trainer.evaluate()
        self.assertTrue(result["eval_loss"] > 0.5)


if __name__ == "__main__":
    unittest.main()
