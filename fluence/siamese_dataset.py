import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import numpy as np
import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset
from transformers import GlueDataTrainingArguments, default_data_collator
from transformers.data.processors.glue import glue_output_modes, glue_processors
from transformers.data.processors.utils import InputExample, InputFeatures
from transformers.tokenization_bart import BartTokenizer, BartTokenizerFast
from transformers.tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_xlm_roberta import XLMRobertaTokenizer

logger = logging.getLogger(__name__)


def siamese_data_collator(batch):
    features_a, features_b = [], []
    for item in batch:
        for k, v in item.items():
            if k == "a":
                features_a.append(v)
            else:
                features_b.append(v)
    return {
        "a": default_data_collator(features_a),
        "b": default_data_collator(features_b),
    }


def siamese_glue_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding_a = tokenizer(
        [example.text_a for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    batch_encoding_b = tokenizer(
        [example.text_b for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    features_a, features_b = [], []
    for i in range(len(examples)):
        inputs_a = {k: batch_encoding_a[k][i] for k in batch_encoding_a}
        inputs_b = {k: batch_encoding_b[k][i] for k in batch_encoding_b}

        feature_a = InputFeatures(**inputs_a, label=labels[i])
        feature_b = InputFeatures(**inputs_b, label=labels[i])
        features_a.append(feature_a)
        features_b.append(feature_b)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features_a[i])

    return [features_a, features_b]


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class SiameseGlueDataset(Dataset):

    args: GlueDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: GlueDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.processor = glue_processors[args.task_name]()
        self.output_mode = glue_output_modes[args.task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                args.task_name,
            ),
        )

        label_list = self.processor.get_labels()
        if args.task_name in ["mnli", "mnli-mm"] and tokenizer.__class__ in (
            RobertaTokenizer,
            RobertaTokenizerFast,
            XLMRobertaTokenizer,
            BartTokenizer,
            BartTokenizerFast,
        ):
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                features = torch.load(cached_features_file)
                self.features_a = features["a"]
                self.features_b = features["b"]
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took"
                    " %.3f s]",
                    time.time() - start,
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args.data_dir)
                else:
                    examples = self.processor.get_train_examples(args.data_dir)
                if limit_length is not None:
                    examples = examples[:limit_length]
                (
                    self.features_a,
                    self.features_b,
                ) = siamese_glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(
                    {"a": self.features_a, "b": self.features_b}, cached_features_file
                )
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]",
                    cached_features_file,
                    time.time() - start,
                )

    def __len__(self):
        return len(self.features_a)

    def __getitem__(self, i) -> InputFeatures:
        return {"a": self.features_a[i], "b": self.features_b[i]}

    def get_labels(self):
        return self.label_list, self.label_list
