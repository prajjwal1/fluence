import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm, trange
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    GlueDataset,
)
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)
from transformers.data.data_collator import DataCollator

from fluence.meta import MetaDataset, MetaTrainer

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": (
                "Path to pretrained model or model identifier from"
                " huggingface.co/models"
            )
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Where do you want to store the pretrained models downloaded from s3"
            )
        },
    )


@dataclass
class MetaArguments(TrainingArguments):
    train_task: Optional[str] = field(
        default=None, metadata={"help": "Support dataset"}
    )
    eval_task: Optional[str] = field(default=None, metadata={"help": "Query dataset"})
    data_dir: Optional[str] = field(default=None)
    inner_learning_rate: float = field(default=2e-5)
    learning_rate: Optional[float] = field(default=2e-5)  # Outer
    max_len: int = field(default=80)
    eval_method: Optional[str] = field(default=None)
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences"
                " longer than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    output_file_name: Optional[str] = field(default="results")


def main():
    parser = HfArgumentParser((ModelArguments, MetaArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not"
            " empty. Use --overwrite_output_dir to overcome."
        )

    # Set seed
    set_seed(training_args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s,"
        " 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    set_seed(training_args.seed)

    try:
        num_labels = glue_tasks_num_labels[training_args.train_task]
        output_mode = glue_output_modes[training_args.train_task]
    except KeyError:
        raise ValueError("Task not found: %s" % (training_args.train_task))

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=training_args.train_task,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction) -> Dict:
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics(training_args.task_name, preds, p.label_ids)

        return compute_metrics_fn

    data_dir = {
        "mrpc": training_args.data_dir + "/MRPC",
        "sst-2": training_args.data_dir + "/SST-2",
        "cola": training_args.data_dir + "/Cola",
        "sts-b": training_args.data_dir + "/STS-B",
    }

    training_args.task_name = training_args.train_task
    training_args.data_dir = data_dir[training_args.task_name]
    train_dataset = GlueDataset(training_args, tokenizer=tokenizer)
    meta_dataset = MetaDataset(train_dataset)
    training_args.task_name = training_args.eval_task
    training_args.data_dir = data_dir[training_args.task_name]
    eval_dataset = GlueDataset(training_args, tokenizer=tokenizer, mode="dev")

    meta_trainer = MetaTrainer(
        model=model,
        args=training_args,
        train_dataset=meta_dataset,
        eval_dataset=eval_dataset,
        train_data_collator=torch.utils.data._utils.collate.default_collate,
        eval_data_collator=default_data_collator,
    )

    meta_trainer.train()


if __name__ == "__main__":
    main()
