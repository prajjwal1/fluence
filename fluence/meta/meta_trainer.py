import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import higher
import pandas as pd
import torch
from torch.optim.sgd import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    GlueDataset,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.data.data_collator import DataCollator
from transformers.trainer import SequentialDistributedSampler
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, is_wandb_available

from .meta_dataset import MetaDataset

logger = logging.getLogger(__name__)


@dataclass
class MetaTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        train_dataset: Optional[MetaDataset] = None,
        eval_dataset: Optional[DataLoader] = None,
        train_data_collator: Optional[DataCollator] = None,
        eval_data_collator: Optional[DataCollator] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        prediction_loss_only=False,
        optimizers: Tuple[
            torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR
        ] = None,
    ):

        self.model = model.to(args.device)
        self.args = args
        self.compute_metrics = compute_metrics
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_data_collator = train_data_collator
        self.eval_data_collator = (
            eval_data_collator if not None else default_data_collator
        )
        self.prediction_loss_only = prediction_loss_only
        self.optimizers = optimizers
        self.eval_results = {}
        set_seed(self.args.seed)

    def get_loss_mean(self, loss):
        return loss.mean() if self.args.n_gpu > 1 else loss

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = (
            RandomSampler(self.train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(self.train_dataset)
        )

        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.train_data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

        return data_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(eval_dataset)
        else:
            sampler = SequentialSampler(eval_dataset)

        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.eval_data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

        return data_loader

    def put_on_device(self, inputs):
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)
        if not isinstance(inputs["labels"], torch.cuda.LongTensor):
            inputs["labels"] = inputs["labels"].long()
        return inputs

    def train(self):
        train_dataloader = self.get_train_dataloader()
        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
        columns = [self.args.train_task, self.args.eval_task]
        metrics = [
            "eval_loss",
            "eval_acc",
            "eval_f1",
            "eval_acc_and_f1",
            "eval_mnli-mm/acc",
        ]
        df = pd.DataFrame(columns=columns, index=metrics)
        for i in range(len(df.columns)):
            for j in range(len(metrics)):
                df[columns[i]][metrics[j]] = []

        model = self.model
        optimizer, scheduler = self.get_optimizers(
            int(
                len(train_dataloader)
                // self.args.gradient_accumulation_steps
                * self.args.num_train_epochs
            )
        )

        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )
        # TODO: Make calculation of num_epochs with HF
        num_train_epochs = self.args.num_train_epochs
        total_train_batch_size = (
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
        )

        logger.info("***** Running training *****")
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info(
            "  Instantaneous batch size per device = %d",
            self.args.per_device_train_batch_size,
        )
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            total_train_batch_size,
        )
        logger.info(
            "  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps
        )

        model.zero_grad()
        self.global_step = 0

        if self.args.eval_method == "every_2":
            eval_step = [2 ** i for i in range(1, 20)]

        inner_optimizer = torch.optim.SGD(
            model.parameters(), lr=self.args.learning_rate
        )
        model.train()

        tqdm_iterator = tqdm(train_dataloader, desc="Batch Index")

        for epoch in tqdm(range(int(self.args.num_train_epochs))):
            if isinstance(train_dataloader, DataLoader) and isinstance(
                train_dataloader.sampler, DistributedSampler
            ):
                train_dataloader.sampler.set_epoch(epoch)

            for batch_idx, meta_batch in enumerate(tqdm_iterator):
                model.zero_grad()
                target_batch = next(iter(eval_dataloader))
                outer_loss = 0.0
                for inputs, target_inputs in zip(meta_batch, target_batch):

                    inputs = self.put_on_device(inputs)
                    target_inputs = self.put_on_device(inputs)

                    with higher.innerloop_ctx(
                        model, inner_optimizer, copy_initial_weights=False
                    ) as (fmodel, diffopt):

                        inner_loss = model(**inputs)[0]
                        inner_loss = self.get_loss_mean(inner_loss)
                        diffopt.step(inner_loss)
                        outer_loss += model(**target_inputs)[0]

                self.global_step += 1
                outer_loss = self.get_loss_mean(outer_loss)
                outer_loss.backward()
                optimizer.step()

                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.args.max_grad_norm
                    )

                # Run evaluation on eval_task
                if self.global_step in eval_step:
                    result = self.evaluate()
                    for key, value in result.items():
                        logger.info(
                            "%s  %s = %s", self.args.eval_task, key, value,
                        )
                    df[self.args.train_task][key].append(value)

                # Save model
                if (
                    self.args.save_steps > 0
                    and self.global_step % self.args.save_steps == 0
                ):
                    if hasattr(model, "module"):
                        assert model.module is self.model
                    else:
                        assert model is self.model

                    output_dir = os.path.join(
                        self.args.output_dir,
                        f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}",
                    )

                    self.save_model(output_dir)
                    if self.is_world_master():
                        self._rotate_checkpoints()

                    logging.info(
                        "*** Results have been saved at %s ***", self.args.output_dir
                    )
                    df.to_csv(
                        self.args.output_dir + self.args.output_file_name + ".csv"
                    )
