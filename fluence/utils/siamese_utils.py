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
from transformers.trainer_utils import PredictionOutput

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

        if self.args.past_index >= 0 and self._past is not None:
            inputs["a"]["mems"] = self._past

        with autocast:
            outputs = model(**inputs)
            loss = outputs[0]
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

    def prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.
        Works both with or without labels.
        """
        if hasattr(self, "_prediction_loop"):
            warnings.warn(
                "The `_prediction_loop` method is deprecated and won't be called in a future version, define `prediction_loop` in your subclass.",
                FutureWarning,
            )
            return self._prediction_loop(dataloader, description, prediction_loss_only=prediction_loss_only)

        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        disable_tqdm = not self.is_local_process_zero() or self.args.disable_tqdm
        samples_count = 0
        for inputs in tqdm(dataloader, desc=description, disable=disable_tqdm):
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only)
            batch_size = inputs["a"][list(inputs.keys())[0]].shape[0]
            samples_count += batch_size
            if loss is not None:
                eval_losses.append(loss * batch_size)
            if logits is not None:
                preds = logits if preds is None else torch.cat((preds, logits), dim=0)
            if labels is not None:
                label_ids = labels if label_ids is None else torch.cat((label_ids, labels), dim=0)

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = self.distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = self.distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))
        elif is_torch_tpu_available():
            # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
            if preds is not None:
                preds = xm.mesh_reduce("eval_preds", preds, torch.cat)
            if label_ids is not None:
                label_ids = xm.mesh_reduce("eval_label_ids", label_ids, torch.cat)

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.sum(eval_losses) / samples_count

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

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
        print(labels)
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
