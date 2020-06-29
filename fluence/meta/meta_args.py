from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from transformers import TrainingArguments


@dataclass
class MetaArguments(TrainingArguments):
    train_task: List = field(default=None, metadata="Support dataset")
    eval_task: List = field(default=None, metadata="Query dataset")
    data_dir: str = field(default=None)
    inner_learning_rate: float = field(default=1e-3)
    outer_learning_rate: float = field(default=2e-5)
    max_len: int = field(default=80)
    eval_method: str = field(default=None)
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
