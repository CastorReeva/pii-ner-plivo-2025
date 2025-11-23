

---

# PII Entity Recognition – Plivo x IIT Madras Assignment

This repository contains my solution for the Plivo + IIT Madras ML assignment.
The goal was to build an end-to-end PII Named Entity Recognition (NER) system using a transformer model, synthetic dataset generation, and span-based evaluation.

---

##  Model Used

* **DistilBERT (Token Classification)**
* Lightweight classifier head on top
* Trained for **3 epochs**
* Max length: **256**
* Batch size: **8**
* Learning rate: **3e-5**

---

##  Key Results

### **Per-Entity F1 Scores**

| Entity      | F1 Score |
| ----------- | -------- |
| CITY        | 1.00     |
| EMAIL       | 1.00     |
| LOCATION    | 1.00     |
| PERSON_NAME | 1.00     |
| DATE        | 0.582    |
| CREDIT_CARD | 0.00     |
| PHONE       | 0.00     |

---

### **PII-only Metrics**

* Precision = **1.00**
* Recall = **1.00**
* F1 = **1.00**

### **Non-PII Metrics**

* Precision = **1.00**
* Recall = **1.00**
* F1 = **1.00**

---

##  Latency (batch = 1, 50 runs)

* p50 = **3.56 ms**
* p95 = **4.15 ms**

---

##  File Structure

```
github_repo/
├── src/
├── data/
├── predictions/
├── model_output/      (Stored in Google Drive)
├── assignment.md
├── requirements.txt
├── metrics.txt
└── README.md
```

---

##  Download Fine-Tuned Model (Google Drive)

GitHub does not allow uploading files >25MB, so the complete trained model is hosted on Google Drive.

📥 **Download Model:**
[https://drive.google.com/file/d/1EOFEH_pelvQWEm72oC_hI8j1WfSKCVd9/view?usp=sharing](https://drive.google.com/file/d/1EOFEH_pelvQWEm72oC_hI8j1WfSKCVd9/view?usp=sharing)

You can load the model using:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_path = "path/to/model_output"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
```

---

##  Running the Project

### **1. Train**

```bash
python src/train.py --train data/train.jsonl --dev data/dev.jsonl --out_dir model_output
```

### **2. Predict**

```bash
python src/predict.py --model_dir model_output --input data/dev.jsonl --output predictions/dev_pred.json
```

### **3. Evaluate**

```bash
python src/eval_span_f1.py --gold data/dev.jsonl --pred predictions/dev_pred.json
```

### **4. Latency Measurement**

```bash
python src/measure_latency.py --model_dir model_output --input data/dev.jsonl
```

---

##  Notes

This model is optimized for:

* Low-latency inference
* Real-time call-center & voice-assistant PII detection
* Robust span extraction
* Easy deployment within distributed systems (aligns with Plivo’s voice AI stack)

---

