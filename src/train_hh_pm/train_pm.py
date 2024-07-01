import os
import re

import torch.distributed as dist
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from trl import RewardConfig, RewardTrainer

accelerator = Accelerator()
os.environ["HF_HOME"] = "/nas/ucb/marcuswilliams/cache/"


def is_main_process():
    return dist.get_rank() == 0


if __name__ == "__main__":
    parser = HfArgumentParser(RewardConfig)
    # Add custom arguments
    parser.add_argument("--model_name", type=str, default=None)  #
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--num_proc", type=int, default=32)
    parser.add_argument("--LoRA", type=str, default="False")
    parser.add_argument("--LoRA_r", type=int, default=None)
    parser.add_argument("--LoRA_alpha", type=int, default=None)
    parser.add_argument("--LoRA_dropout", type=float, default=None)
    parser.add_argument("--margin", type=str, default="True")
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--perspective", type=str, default="3_1")
    # parser.add_argument("--bf16", type=bool, default=True)
    # Parse the dictionary into RewardConfig
    reward_config, config = parser.parse_args_into_dataclasses()

    # print(reward_config)f
    # reward_config.gradient_checkpointing_kwargs={"use_reentrant":False}
    train_dataset = load_dataset("json", data_files=f"data/hh_labels/hh_train_{config.perspective}.jsonl")["train"]
    test_dataset = load_dataset("json", data_files=f"data/hh_labels/hh_test_{config.perspective}.jsonl")["train"]
    # train_dataset = load_dataset("json", data_files=f"data/hh_labels/anthropic_train.jsonl")["train"]
    # test_dataset = load_dataset("json", data_files=f"data/hh_labels/anthropic_test.jsonl")["train"]

    # train_dataset = train_dataset.select(range(5))
    test_dataset = test_dataset.select(range(1024))

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=config.LoRA_r,
        lora_alpha=config.LoRA_alpha,
        lora_dropout=config.LoRA_dropout,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name, use_fast=True, padding="max_length", max_length=reward_config.max_length, truncation=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=1)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    if getattr(model.config, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    def format_prompt(text):

        pattern = r"\n\nAssistant:|\n\nHuman:"

        conversation = re.split(pattern, text)[1:]
        messages = []
        for i, e in enumerate(conversation):
            if i % 2 == 0:
                messages.append({"role": "user", "content": e.strip()})
            else:
                messages.append({"role": "assistant", "content": e.strip()})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False).replace(
            tokenizer.bos_token, ""
        )

    def preprocess_func(examples):
        inputs_chosen, inputs_rejected = [], []
        attention_masks_chosen, attention_masks_rejected = [], []
        margins = []
        for i in range(len(examples["chosen"])):
            chosen_prompt = format_prompt(examples["chosen"][i])
            rejected_prompt = format_prompt(examples["rejected"][i])
            # chosen_prompt = examples["chosen"][i]
            # rejected_prompt = examples["rejected"][i]
            logits_chosen = float(examples["logits_chosen"][i])
            logits_rejected = float(examples["logits_rejected"][i])

            tokenized_chosen = tokenizer(
                chosen_prompt, truncation="longest_first", padding="longest", max_length=reward_config.max_length
            )
            tokenized_rejected = tokenizer(
                rejected_prompt, truncation="longest_first", padding="longest", max_length=reward_config.max_length
            )
            if logits_chosen > logits_rejected:
                tokenized_chosen = tokenizer(
                    chosen_prompt, truncation="longest_first", padding="longest", max_length=reward_config.max_length
                )
                tokenized_rejected = tokenizer(
                    rejected_prompt, truncation="longest_first", padding="longest", max_length=reward_config.max_length
                )
                margin = logits_chosen - logits_rejected
            else:
                tokenized_chosen = tokenizer(
                    rejected_prompt, truncation="longest_first", padding="longest", max_length=reward_config.max_length
                )
                tokenized_rejected = tokenizer(
                    chosen_prompt, truncation="longest_first", padding="longest", max_length=reward_config.max_length
                )
                margin = logits_rejected - logits_chosen
            inputs_chosen.append(tokenized_chosen["input_ids"])
            attention_masks_chosen.append(tokenized_chosen["attention_mask"])
            inputs_rejected.append(tokenized_rejected["input_ids"])
            attention_masks_rejected.append(tokenized_rejected["attention_mask"])
            margins.append(margin)

        d = {
            "input_ids_chosen": inputs_chosen,
            "attention_mask_chosen": attention_masks_chosen,
            "input_ids_rejected": inputs_rejected,
            "attention_mask_rejected": attention_masks_rejected,
        }
        if config.margin == "True":
            d["margin"] = margins
        return d

    # preprocess the dataset
    train_dataset = train_dataset.map(
        preprocess_func,
        batched=True,
        num_proc=config.num_proc,
    )
    test_dataset = test_dataset.map(
        preprocess_func,
        batched=True,
        num_proc=config.num_proc,
    )
    print("begin training")
    model.train()

    # reward_config.bf16 = True
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model()
    model = accelerator.unwrap_model(model)
    model.save_pretrained(reward_config.output_dir + "/unwrap")
