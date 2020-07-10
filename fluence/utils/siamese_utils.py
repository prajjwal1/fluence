import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
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

    def _training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Dict[str, Union[torch.Tensor, Any]]],
        optimizer: torch.optim.Optimizer,
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

        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()

    def _prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.
        Works both with or without labels.
        """

        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else self.prediction_loss_only
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
            past = None

        for inputs in tqdm(dataloader, desc=description):
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

            if self.args.past_index >= 0:
                inputs["a"]["mems"] = past

            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]
                if self.args.past_index >= 0:
                    past = outputs[
                        self.args.past_index if has_labels else self.args.past_index - 1
                    ]

            if not prediction_loss_only:
                if preds is None:
                    preds = logits.detach()
                else:
                    preds = torch.cat((preds, logits.detach()), dim=0)
                if inputs["a"].get("labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["a"]["labels"].detach()
                    else:
                        label_ids = torch.cat(
                            (label_ids, inputs["a"]["labels"].detach()), dim=0
                        )

        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = self.distributed_concat(
                    preds, num_total_examples=self.num_examples(dataloader)
                )
            if label_ids is not None:
                label_ids = self.distributed_concat(
                    label_ids, num_total_examples=self.num_examples(dataloader)
                )
        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = preds.cpu().numpy()

        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()

        if (
            self.compute_metrics is not None
            and preds is not None
            and label_ids is not None
        ):
            print(len(preds), len(label_ids))
            metrics = self.compute_metrics(
                EvalPrediction(predictions=preds, label_ids=label_ids)
            )
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

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
