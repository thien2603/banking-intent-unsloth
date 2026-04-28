#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference class for banking intent classification.
"""

import yaml
import torch
import re
from unsloth import FastLanguageModel

class IntentClassification:
    def __init__(self, model_path):
        with open(model_path, "r") as f:
            config = yaml.safe_load(f)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=config["model_checkpoint"],
            max_seq_length=config.get("max_seq_length", 512),
            load_in_4bit=config.get("load_in_4bit", True),
        )
        FastLanguageModel.for_inference(self.model)
    
    def call(self, message):
        # Định dạng prompt giống như khi train
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nIntent:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=5, do_sample=False)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Trích xuất số nguyên sau "Intent:"
        match = re.search(r"Intent:\s*(\d+)", response)
        if match:
            return match.group(1)
        else:
            return "unknown"

if __name__ == "__main__":
    # Ví dụ sử dụng
    classifier = IntentClassification("configs/inference.yaml")
    test_msg = "I am still waiting on my card?"
    result = classifier.call(test_msg)
    print(f"Message: {test_msg}\nPredicted intent: {result}")