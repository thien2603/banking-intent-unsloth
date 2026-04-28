import os
os.environ["HF_HOME"] = "D:\\huggingface_cache"

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

adapter_path = r"D:\MyStudy\banking-intent-unsloth\final_model\final_model"
base_model_name = "unsloth/Llama-3.2-1B"
max_seq_length = 256

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name, model_max_length=max_seq_length)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model (this may take a few minutes on first run)...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32,
    device_map="cpu",          # hoặc "auto" nếu có RAM lớn
    low_cpu_mem_usage=True,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

def predict_intent(text):
    # Sử dụng chat template đúng với Llama-3.2 (có thể điều chỉnh)
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nIntent:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_length)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    match = re.search(r"Intent:\s*(\d+)", response)
    return match.group(1) if match else "unknown"

if __name__ == "__main__":
    queries = [
        "I am still waiting on my card?",
        "What is the exchange rate for USD to EUR?",
        "I lost my credit card, please help!"
    ]
    for q in queries:
        intent_id = predict_intent(q)
        print(f"Q: {q}\nIntent ID: {intent_id}\n")