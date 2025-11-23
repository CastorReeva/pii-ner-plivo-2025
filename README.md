
# PII Entity Recognition - Plivo x IIT Madras Assignment

This repository contains my solution for the Plivo + IIT Madras ML assignment.
The goal was to build an end-to-end PII Named Entity Recognition (NER) system using deep learning, custom dataset generation, and span-based evaluation.

## Model Used
- DistilBERT Token Classification
- Lightweight classifier head
- Fine-tuned for 3 epochs
- Max length: 256
- Batch size: 8
- Learning rate: 3e-5

## Key Results

### Per-Entity F1
CITY = 1.00
EMAIL = 1.00
LOCATION = 1.00
PERSON_NAME = 1.00
DATE = 0.582
CREDIT_CARD = 0
PHONE = 0

### PII-only Metrics:
Precision = 1.00
Recall = 1.00
F1 = 1.00

### Non-PII Metrics:
Precision = 1.00
Recall = 1.00
F1 = 1.00

### Latency (batch=1, 50 runs):
p50 = 3.56 ms
p95 = 4.15 ms

## File Structure
github_repo/
├── src/
├── data/
├── predictions/
├── model_output/
├── requirements.txt
├── assignment.md
└── README.md

## Running the Project

### Train
python src/train.py --train data/train.jsonl --dev data/dev.jsonl --out_dir model_output

### Predict
python src/predict.py --model_dir model_output --input data/dev.jsonl --output predictions/dev_pred.json

### Evaluate
python src/eval_span_f1.py --gold data/dev.jsonl --pred predictions/dev_pred.json

### Latency
python src/measure_latency.py --model_dir model_output --input data/dev.jsonl

## Notes
This model is optimized for low latency and real-time use cases such as voice assistants and call-center AI systems, aligning with Plivo's engineering stack.
