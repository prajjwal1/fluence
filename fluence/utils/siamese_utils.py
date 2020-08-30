import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import EvalPrediction, Trainer
from transformers.file_utils import is_apex_available, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput

if is_apex_available():
    from apex import amp

logger = logging.getLogger(__name__)


@dataclass
class SiameseModelArguments:
    """
    Arguments pertaining to SiameseTransformer
    """

    model_name: str = field(
        metadata={
            "help": (
                "Path to pretrained model or model identifier from"
                " huggingface.co/models"
            )
        }
    )
    load_model_path: str = field(
        default=None, metadata={"help": "Path from where weights will be loaded"}
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


class SiameseTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Dict[str, Union[torch.Tensor, Any]]],
    ) -> float:
        model.train()
        for k, v in inputs["a"].items():
            if isinstance(v, torch.Tensor):
                inputs["a"][k] = v.to(self.args.device)

        for k, v in inputs["b"].items():
            if isinstance(v, torch.Tensor):
                inputs["b"][k] = v.to(self.args.device)
        # for k, v in inputs.items():
        #    if isinstance(v, torch.Tensor):
        #        inputs[k] = v.to(self.args.device)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["a"]["mems"] = self._past
            # inputs["mems"] = self._past

        with autocast:
            outputs = model(**inputs)
            loss = outputs[
                0
            ]  # model outputs are always tuple in transformers (see doc)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item()

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """
        has_labels = any(
            inputs["a"].get(k) is not None
            for k in ["labels", "lm_labels", "masked_lm_labels"]
        )

        for k, v in inputs["a"].items():
            if isinstance(v, torch.Tensor):
                inputs["a"][k] = v.to(self.args.device)

        for k, v in inputs["b"].items():
            if isinstance(v, torch.Tensor):
                inputs["b"][k] = v.to(self.args.device)

        with torch.no_grad():
            outputs = model(**inputs)
            if has_labels:
                loss, logits = outputs[:2]
                loss = loss.mean().item()
            else:
                loss = None
                logits = outputs[0]
            if self.args.past_index >= 0:
                self._past = outputs[
                    self.args.past_index if has_labels else self.args.past_index - 1
                ]

        if prediction_loss_only:
            return (loss, None, None)

        labels = inputs["a"].get("labels")
        if labels is not None:
            labels = labels.detach()
        return (loss, logits.detach(), labels)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(
            {"model_state_dict": self.model.state_dict()},
            os.path.join(output_dir, "pytorch_model.bin"),
        )
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
