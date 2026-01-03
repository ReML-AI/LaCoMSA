# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# parts of this code is adapted from the MAPO paper

import os
import shutil

# 0. imports
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, Optional

import deepspeed
import pandas as pd
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from trl import DPOConfig, DPOTrainer

Log_dir = "../Log"
os.environ["WANDB_PROJECT"] = "lacomsa"

from transformers import modeling_utils

if (
    not hasattr(modeling_utils, "ALL_PARALLEL_STYLES")
    or modeling_utils.ALL_PARALLEL_STYLES is None
):
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        if args.local_rank == 0:
            dir_lst = os.listdir(checkpoint_folder)
            global_step_path = ""
            for dir_ele in dir_lst:
                if "global_step" in dir_ele:
                    global_step_path = os.path.join(checkpoint_folder, dir_ele)
            if os.path.exists(global_step_path):
                shutil.rmtree(global_step_path)
        return control


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    training_config: Optional[str] = field(
        default=None, metadata={"help": "Path to training config"}
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata={"help": "the lr scheduler type"}
    )
    warmup_steps: Optional[int] = field(
        default=100, metadata={"help": "the number of warmup steps"}
    )
    weight_decay: Optional[float] = field(
        default=0.05, metadata={"help": "the weight decay"}
    )
    # optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
    optimizer_type: Optional[str] = field(
        default="adamw_torch", metadata={"help": "the optimizer type"}
    )

    per_device_train_batch_size: Optional[int] = field(
        default=2, metadata={"help": "train batch size per device"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=2, metadata={"help": "eval batch size per device"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=8, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    max_prompt_length: Optional[int] = field(
        default=512, metadata={"help": "the maximum prompt length"}
    )
    max_length: Optional[int] = field(
        default=1024, metadata={"help": "the maximum sequence length"}
    )
    max_steps: Optional[int] = field(
        default=500, metadata={"help": "max number of training steps"}
    )
    logging_steps: Optional[int] = field(
        default=10, metadata={"help": "the logging frequency"}
    )
    save_steps: Optional[int] = field(
        default=100, metadata={"help": "the saving frequency"}
    )
    eval_steps: Optional[int] = field(
        default=2000, metadata={"help": "the evaluation frequency"}
    )

    # instrumentation
    sanity_check: Optional[bool] = field(
        default=False, metadata={"help": "only train on 1000 samples"}
    )
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    rpo_alpha: Optional[float] = field(
        default=None,
        metadata={
            "help": "the alpha value for RPO loss, 1.0 is the default value in the paper"
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1, metadata={"help": "the number of training epochs"}
    )
    seed: Optional[int] = field(
        default=42, metadata={"help": "the random seed for reproducibility"}
    )
    max_training_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "the maximum number of training samples to use, if None, use all samples"
        },
    )


def get_hh(args) -> Dataset:
    try:
        datasets_list = [
            # load_dataset("json", data_files=i + "-train.json")["train"]
            Dataset.from_pandas(pd.read_json(i + "-train.json"))
            for i in args.dataset
        ]
        train_dataset = concatenate_datasets(datasets_list).shuffle(seed=42)

        try:
            datasets_list = [
                # load_dataset("json", data_files=i + "-dev.json")["train"] for i in args.dataset
                Dataset.from_pandas(pd.read_json(i + "-dev.json"))
                for i in args.dataset
            ]
            dev_dataset = concatenate_datasets(datasets_list).shuffle(seed=42)
        except Exception as e:
            datasets_list = [
                # load_dataset("json", data_files=i + "-dev.json")["train"] for i in args.dataset
                Dataset.from_pandas(pd.read_json(i + "-train.json"))
                for i in args.dataset
            ]
            dev_dataset = concatenate_datasets(datasets_list).shuffle(seed=42)
    except Exception as e:
        ds = load_dataset(args.dataset[0])
        train_dataset = ds["train"]
        dev_dataset = ds["test"]

    if len(dev_dataset) > 2_000:  # 20_000
        dev_dataset = dev_dataset.select(range(2_000))

    if args.max_training_samples and len(train_dataset) > args.max_training_samples:
        train_dataset = train_dataset.select(range(args.max_training_samples))

    return train_dataset, dev_dataset


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    import json

    f = open("./dpo_config/" + script_args.training_config, "r")
    training_details = json.load(f)
    script_args.model_name_or_path = training_details["model_name_or_path"]
    script_args.dataset = training_details["dataset"]
    script_args.beta = training_details["beta"]
    script_args.learning_rate = training_details["learning_rate"]
    # script_args.max_steps = training_details["max_step"]
    script_args.gradient_accumulation_steps = training_details[
        "gradient_accumulation_steps"
    ]
    script_args.per_device_train_batch_size = training_details[
        "per_device_train_batch_size"
    ]
    script_args.per_device_eval_batch_size = training_details[
        "per_device_eval_batch_size"
    ]
    script_args.max_length = training_details["max_length"]
    script_args.output_dir = "../output_model_" + script_args.training_config
    script_args.max_training_samples = training_details["max_training_samples"]
    if script_args.max_training_samples:
        script_args.output_dir += "_max_samples_" + str(
            script_args.max_training_samples
        )
    script_args.warmup_steps = training_details.get("warmup_steps", 20)
    script_args.eval_steps = training_details.get("eval_steps", 2000)
    script_args.rpo_alpha = training_details.get("rpo_alpha", None)
    script_args.num_train_epochs = training_details.get("num_train_epochs", 1)
    script_args.seed = training_details.get("seed", 42)

    # from: https://github.com/huggingface/accelerate/issues/314
    # Create the custom configuration
    process_group_kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=5400)
    )  # 1.5 hours

    # Instantiate Accelerator with the custom configuration
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model_ref.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
    )
    tokenizer.padding_side = "right"  # Allow batched inference

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, eval_dataset = get_hh(script_args)

    # 4. initialize training arguments:
    training_args = DPOConfig(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        # max_steps=script_args.max_steps,
        num_train_epochs=script_args.num_train_epochs,
        logging_steps=script_args.logging_steps,
        # save_steps=script_args.save_steps,
        save_strategy="epoch",
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        eval_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        save_total_limit=3,
        # load_best_model_at_end=True,
        logging_dir=Log_dir + script_args.output_dir.split("/")[-1] + "/logs",
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        # optim='adamw_torch',
        bf16=True,
        # tf32=True,
        remove_unused_columns=True,
        max_grad_norm=1,
        beta=script_args.beta,
        max_length=script_args.max_length,
        rpo_alpha=script_args.rpo_alpha,
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        # max_prompt_length=script_args.max_prompt_length,
        # callbacks=[SavePeftModelCallback],
    )

    dpo_trainer.train()
