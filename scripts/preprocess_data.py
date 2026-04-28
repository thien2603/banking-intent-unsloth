#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tune a model on BANKING77 dataset using Unsloth with QLoRA.
"""

import pandas as pd
import torch
from datasets import Dataset
from transformers import TrainingArguments
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
import yaml

def main():
    with open("configs/train.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load model với 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
    )
    
    # Thêm LoRA adapter
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        target_modules=config["lora_target_modules"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.get("random_state", 42),
    )
    
    # Đọc dữ liệu đã tiền xử lý
    train_df = pd.read_csv("sample_data/train.csv")
    val_df = pd.read_csv("sample_data/val.csv")
    
    # Format prompt theo chuẩn Llama-3
    def format_func(examples):
        texts = []
        for text, label in zip(examples["text"], examples["label_text"]):
            prompt = (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                f"Intent: {label}<|eot_id|>"
            )
            texts.append(prompt)
        return {"text": texts}
    
    train_dataset = Dataset.from_pandas(train_df[["text", "label_text"]]).map(format_func, batched=True)
    val_dataset = Dataset.from_pandas(val_df[["text", "label_text"]]).map(format_func, batched=True)
    
    # Cấu hình trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=TrainingArguments(
            output_dir=config["output_dir"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config.get("eval_batch_size", config["batch_size"]),
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            warmup_steps=config.get("warmup_steps", 10),
            max_steps=config.get("max_steps", -1),
            num_train_epochs=config.get("num_epochs", 3) if config.get("max_steps", -1) == -1 else None,
            learning_rate=config["learning_rate"],
            logging_steps=config.get("logging_steps", 10),
            save_steps=config.get("save_steps", 100),
            eval_strategy="steps",
            eval_steps=config.get("eval_steps", 50),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim=config.get("optim", "adamw_8bit"),
            weight_decay=config.get("weight_decay", 0.01),
            lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
            seed=config.get("seed", 3407),
            report_to="none",
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
    )
    
    # Bắt đầu training
    trainer.train()
    
    # Lưu model và tokenizer
    model.save_pretrained("outputs/final_model")
    tokenizer.save_pretrained("outputs/final_model")
    print("Done! Model saved at outputs/final_model")

if __name__ == "__main__":
    main()