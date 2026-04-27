# 📊 Banking Intent Classification with Unsloth

## 📌 Overview

This project fine-tunes a model using Unsloth (QLoRA) for intent classification on a subset of the BANKING77 dataset. The work follows the requirements of Project 2 – NLP in Industry.

---

## 📊 Data Preparation

**Dataset:** BANKING77 (original: 13,083 queries, 77 intent classes)

A subset is sampled to fit available computational resources.

### 🔧 Preprocessing steps (`scripts/preprocess_data.py`)

* Load dataset from Hugging Face
* Sample a subset (e.g., 5,000 train / 1,000 test)
* Basic text cleaning and normalization
* Map intent labels to IDs
* Split into train / test (and optionally validation)
* Save processed data to:

  * `sample_data/train.csv`
  * `sample_data/test.csv`

### ▶️ Run preprocessing

```bash id="p1x8ad"
python scripts/preprocess_data.py
```

---

## 🤖 Model Fine-tuning with Unsloth

* **Base model:** Any encoder supported by Unsloth
  (e.g., `unsloth/Llama-3.2-1B`, `answerdotai/ModernBERT-base`)
* **Method:** QLoRA (4-bit quantization + LoRA adapters)

---

### ⚙️ Hyperparameters (`configs/train.yaml`)

| Parameter             | Typical value |
| --------------------- | ------------- |
| LoRA rank (r)         | 16            |
| LoRA alpha            | 16            |
| LoRA dropout          | 0.0           |
| Batch size            | 8 – 16        |
| Gradient accumulation | 1 – 2         |
| Learning rate         | 2e-4          |
| Optimizer             | adamw_8bit    |
| Epochs                | 3             |
| Max sequence length   | 256           |
| Warmup steps          | 10            |

*(Adjust according to your actual training configuration.)*

---

### ▶️ Run training

```bash id="8n6lqk"
python scripts/train.py
```

Training saves the best checkpoint to `outputs/`.

---

## 🎯 Inference

The inference class is implemented in `scripts/inference.py` and follows the required interface:

* `__init__(self, model_path)` – loads config, tokenizer, and model checkpoint
* `call(self, message)` – returns predicted intent label as string

---

### 💡 Example usage

```python id="k2q4zx"
from scripts.inference import IntentClassification

classifier = IntentClassification("configs/inference.yaml")
pred = classifier.call("I am still waiting on my card?")
print(pred)  # e.g., 'card_arrival'
```

---

### ▶️ Run inference

```bash id="r5d9wj"
python scripts/inference.py
```

---

## 📈 Results (to be filled after actual training)

Test accuracy, training time, and VRAM usage will be updated after running the code.

---

## 🎥 Video Demonstration

📁 Google Drive folder:
https://drive.google.com/drive/folders/11OM-9Bb6sRR80MKCYbdPfHRsVbJHhQrg?usp=drive_link

### The video demonstrates:

* Running the inference script
* Example input messages and predicted intents
* Final test accuracy (displayed during evaluation)

---

## 📁 Repository Structure

```id="z7h3nb"
banking-intent-unsloth/
├── scripts/
│   ├── train.py
│   ├── inference.py
│   └── preprocess_data.py
├── configs/
│   ├── train.yaml
│   └── inference.yaml
├── sample_data/
│   ├── train.csv
│   └── test.csv
├── outputs/               (saved model checkpoints)
├── train.sh
├── inference.sh
├── requirements.txt
└── README.md
```

---

## 📚 References

* BANKING77 Dataset
* Unsloth Documentation

---

## 📬 Contact

* **Student:** Trần Danh Thiện
* **GitHub:** https://github.com/thien2603
* **Project Repository:** https://github.com/thien2603/banking-intent-unsloth
