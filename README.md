# Toxic Comment Classification - MLOps Project

This project implements an end-to-end **MLOps pipeline** for multi-label toxic comment classification using a BERT-based model. 
It includes experiment tracking, model registry, workflow orchestration, model deployment, and monitoring.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Setup & Usage](#setup--usage)
- [Cloud Deployment](#cloud-deployment)
- [Project Structure](#project-structure)
- [Data Processing](#data-processing)
- [Model Selection](#model-selection)
- [Experiment Tracking](#experiment-tracking)
- [Model Deployment](#model-deployment)
- [Orchestration](#orchestration)
- [Monitoring](#monitoring)
- [Code Quality & Automation](#code-quality--automation)
- [Potential Improvements](#potential-improvements)

---

## Problem Statement
Toxic language online poses significant challenges for social platforms, requiring automated tools to identify and filter harmful content.  
This project builds an **AI-powered toxic comment classifier** that detects six toxicity types:
- toxic
- severe toxic
- obscene
- threat
- insult
- identity hate

The goal is to **deploy** a production-ready classifier with a full MLOps workflow for training, deployment, and monitoring.

---

## Setup & Usage

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Git

### Installation
```bash
git clone <your-repo-url>
cd toxic-comment-classifier-mlops
docker-compose up -d
```

### Services
| Service       | URL                                |
|---------------|------------------------------------|
| MLflow        | http://localhost:5050              |
| Prefect UI    | http://localhost:4200              |
| Grafana       | http://localhost:3000              |
| Hugging Face Space | [Live App](https://huggingface.co/spaces/MonaHamid/bert-toxic-classifier) |

---

## Cloud Deployment
The model is deployed on **Hugging Face Spaces** with Gradio for real-time inference.

---

## Project Structure
```
.
├── orchestration/        # Prefect training & monitoring flows
├── scripts/              # Model training scripts
├── src/                  # Preprocessing and constants
├── gradio_app/           # Gradio deployment app
├── docker-compose.yml    # Full MLOps stack
└── README.md
```

---

## Data Processing
- Source: Kaggle Jigsaw Toxic Comment Classification dataset
- Preprocessing: text cleaning, tokenization for BERT
- Multi-label binarization for six toxicity labels

---

## Model Selection
- **BERT** fine-tuned on toxic comment dataset
- Baselines: Logistic Regression, Naive Bayes, SVM
- Evaluation: F1-micro, F1-macro, per-label F1

---

## Experiment Tracking
- MLflow tracks all experiments, metrics, and artifacts
- Model registry used to manage versions and promote to Production

---

## Model Deployment
- BERT model served via **Gradio** on Hugging Face Spaces
- UI for entering comments and viewing toxicity predictions

---

## Orchestration
- Prefect flows:
  - **Training**: runs model training and logs to MLflow
  - **Monitoring**: checks for drift and missing values

---

## Monitoring
- Drift metrics: `drift_score`, `missing_share`, `drifted_features`
- Stored in PostgreSQL and visualized in Grafana
- Alerts can be configured for threshold violations

---

## Code Quality & Automation
Implemented best practices:
- Black (code formatter)
- isort (import sorter)
- pylint (linting)
- pytest (unit/integration tests)
- pre-commit hooks
- Makefile for common tasks
- GitHub Actions CI/CD pipeline

---

## Potential Improvements
- Add retraining triggers when drift exceeds thresholds
- Integrate Slack alerts
- Deploy monitoring dashboard to cloud
- Advanced hyperparameter tuning for BERT
