# 📊 Banking Intent Classification with Unsloth

## 📌 Overview

This project fine-tunes a modern BERT-style model using QLoRA for intent classification on the BANKING77 dataset. The goal is to achieve strong performance with limited computational resources.

---

## 📊 Data Preparation

The **BANKING77 dataset** consists of 13,083 customer service queries labeled into 77 intent classes (e.g., `card_arrival`, `exchange_rate`, `lost_card`).

To optimize training efficiency, a representative subset was sampled.

### 🔧 Preprocessing Steps (`scripts/preprocess_data.py`)

* **Load dataset** – Using Hugging Face `datasets` library.
* **Sampling**

  * 5,000 training samples
  * 500 validation samples
  * 1,000 test samples
* **Text normalization** – Lowercasing and basic punctuation normalization.
* **Label mapping** – Preserved original indices (0–76).
* **Train/validation/test split** – 70 / 10 / 20 ratio.
* **Save processed data** – Stored as CSV files:

  * `sample_data/train.csv`
  * `sample_data/test.csv`

### ▶️ Run preprocessing

```bash
python scripts/preprocess_data.py
```

---

## 🤖 Model Fine-tuning with Unsloth

### 📌 Model Selection

* **Base model:** `answerdotai/ModernBERT-base` – A modern encoder optimized for classification tasks.

---

### ⚙️ Hyperparameters (`configs/train.yaml`)

| Hyperparameter        | Value                                     | Explanation                      |
| --------------------- | ----------------------------------------- | -------------------------------- |
| Base model            | answerdotai/ModernBERT-base               | Classification-optimized encoder |
| LoRA rank (r)         | 16                                        | Adapter capacity                 |
| LoRA alpha            | 16                                        | Scaling factor                   |
| LoRA dropout          | 0.0                                       | Disabled for faster training     |
| Target modules        | ["query", "key", "value", "output.dense"] | Attention layers                 |
| Batch size            | 16                                        | Per-device                       |
| Gradient accumulation | 2                                         | Effective batch size = 32        |
| Learning rate         | 2e-4                                      | Standard for QLoRA               |
| Optimizer             | adamw_8bit                                | Memory-efficient                 |
| Epochs                | 3                                         | Sufficient for dataset size      |
| Max sequence length   | 256                                       | Covers ~99% of queries           |
| Precision             | 4-bit QLoRA                               | Reduced VRAM usage               |
| Warmup steps          | 10                                        | Stabilizes early training        |
| Logging steps         | 1                                         | Detailed logging                 |

---

### ▶️ Run training

```bash
python scripts/train.py
```

---

### 🧠 Training Pipeline

* Load preprocessed data
* Configure model with QLoRA via Unsloth
* Train and validate model
* Save best checkpoint to `outputs/`
* Select best model based on validation accuracy

---

## 🎯 Inference

The inference interface is implemented in `scripts/inference.py`.

### 🧩 Class Interface

* `__init__(self, model_path)` – Loads config, tokenizer, and model.
* `call(self, message)` – Returns predicted intent label.

---

### 💡 Example Usage

```python
from scripts.inference import IntentClassification

classifier = IntentClassification("configs/inference.yaml")

message = "I am still waiting on my card?"
predicted_label = classifier.call(message)

print(f"Message: {message}\nPredicted intent: {predicted_label}")
```

---

### 🧾 Sample Output

```
Message: I am still waiting on my card?
Predicted intent: card_arrival
```

---

### ▶️ Run inference

```bash
python scripts/inference.py
```

---

## 📈 Results

| Metric        | Value                     |
| ------------- | ------------------------- |
| Test Accuracy | **85.6%** (1,000 samples) |
| Training Time | ~15 minutes (T4 GPU)      |
| VRAM Usage    | ~5 GB (4-bit QLoRA)       |

---

## 🎥 Video Demonstration

👉 Google Drive Link:
https://drive.google.com/drive/folders/11OM-9Bb6sRR80MKCYbdPfHRsVbJHhQrg?usp=drive_link

### The video includes:

* Running inference script
* Multiple prediction examples
* Final test accuracy demonstration

---

## 📁 Repository Structure

```
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
├── outputs/
├── train.sh
├── inference.sh
├── requirements.txt
└── README.md
```

---

## 📚 References

* BANKING77 Dataset – PolyAI
* Unsloth Documentation
* LoRA / QLoRA Best Practices

---

## 📬 Contact

* **Student:** Trần Danh Thiện
* **GitHub:** https://github.com/thien2603
* **Project Repository:** https://github.com/thien2603/banking-intent-unsloth
