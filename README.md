PII Entity Recognition - Plivo x IIT Madras Assignment

This repository contains my solution for the Plivo + IIT Madras ML assignment.
The goal was to build an end-to-end PII Named Entity Recognition (NER) system using deep learning, custom dataset generation, and span-based evaluation.

Model Used

DistilBERT Token Classification

Lightweight classifier head

Fine-tuned for 3 epochs

Max length: 256

Batch size: 8

Learning rate: 3e-5

Download Trained Model (Important)

GitHub does not allow uploading large model files (>25MB).
The complete fine-tuned model has been uploaded to Google Drive.

Download Model:
https://drive.google.com/file/d/1EOFEH_pelvQWEm72oC_hI8j1WfSKCVd9/view?usp=sharing

The file includes:

config.json

model.safetensors

tokenizer.json

tokenizer_config.json

special_tokens_map.json

vocab.txt

How to load:

from transformers import AutoTokenizer, AutoModelForTokenClassification

model_path = "path/to/downloaded/model_output"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

Key Results:

Per-Entity F1 Scores
CITY          = 1.00
EMAIL         = 1.00
LOCATION      = 1.00
PERSON_NAME   = 1.00
DATE          = 0.582
CREDIT_CARD   = 0.00
PHONE         = 0.00

PII-only Metrics

Precision = 1.00
Recall = 1.00
F1 = 1.00

Non-PII Metrics

Precision = 1.00
Recall = 1.00
F1 = 1.00

Latency (batch = 1, 50 runs)

p50 = 3.56 ms
p95 = 4.15 ms

File Structure
github_repo/
├── src/
├── data/
├── predictions/
├── model_output/   (Stored in Google Drive)
├── requirements.txt
├── assignment.md
├── metrics.txt
└── README.md

Running the Project
Train
python src/train.py --train data/train.jsonl --dev data/dev.jsonl --out_dir model_output

Predict
python src/predict.py --model_dir model_output --input data/dev.jsonl --output predictions/dev_pred.json


Evaluate
python src/eval_span_f1.py --gold data/dev.jsonl --pred predictions/dev_pred.json

Latency Measurement
python src/measure_latency.py --model_dir model_output --input data/dev.jsonl

Notes
This model is optimized for:

Low latenc
Real-time inference
Call-center & voice assistant PII detection


