from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,        # hiệu dụng batch = 4*2=8
        warmup_steps=10,
        max_steps=600,                         # đủ để hội tụ
        learning_rate=2e-4,                    # tối ưu cho LoRA
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        eval_strategy="steps",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

print("🚀 Bắt đầu fine-tune (khoảng 20-30 phút)...")
trainer.train()

model.save_pretrained("final_model")
tokenizer.save_pretrained("final_model")
print("✅ Đã lưu model vào thư mục 'final_model'")