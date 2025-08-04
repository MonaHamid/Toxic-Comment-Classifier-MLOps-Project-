# Toxic Comment Classification - MLOps Project

This project implements an end-to-end **MLOps pipeline** for multi-label toxic comment classification using a BERT-based model. 
It includes experiment tracking, model registry, workflow orchestration, model deployment, and monitoring.
<img width="1024" height="1024" alt="e3e484d1-563a-477b-a7c1-fd4508da963b" src="https://github.com/user-attachments/assets/3800fb19-eac4-45d9-9782-71e3d41bb14a" />

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
TIntroduction
The work carried out in this notebook focused on the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) hosted on [Kaggle](https://www.kaggle.com/). The task consists of a **Multilabel text classification** problem where a given toxic comment, needs to be classified into one or more categories out of the following list:
- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

For instance, if the comment is `toxic` and `obscene`, then for both those headers the value will be `1` and for the others it will be `0`.

# Data

- We are using the [Jigsaw toxic data](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data) prived by the competition
- We are referring only to the following csv files from the data dump: `train.csv`, `test.csv`, `test_labels.csv`
- Now that the competition has ended, the labels for the test set have been provided. This will enable us to conduct inference and assess the performance of each tested model.

  # Models
I employed three distinct models using TensorFlow to address the challenge:
- **MODEL I**: a baseline approach that utilized a **logistic regression**.
- **MODEL II**: a variation of the baseline approach that incorporated **Glove's pre-trained embeddings** with the Bidirectional LSTM architecture.
- **MODEL III**: the well-known **BERT model**, which is capable of producing state-of-the-art results on a range of NLP tasks including text classification.

By utilizing these three models, I aimed to determine which approach would yield the most effective results for this particular task.

# **Evaluation metric**
For the evaluation of my models in the Toxic Comment Challenge competition, I opted to use the **macro-averaged F1 score** as the reference metric, rather than AUC-ROC as recommended by the competition guidelines. I believe F1 to be a more appropriate metric for the following reasons: 
- the F1 score is a more comprehensive metric that takes both precision and recall into account and provides a more nuanced assessment of the model's performance across all the different classes. This is particularly important in the Toxic Comment Challenge, where identifying and categorizing multiple types of toxic comments is crucial.
- Moreover, the macro-averaged F1 score does not take class imbalance into account, which ensures that every class is given equal weight independently of its proportion. This is important in imbalanced datasets where one or more classes may be significantly underrepresented.
- In contrast, AUC-ROC can be misleading in case of imbalanced data, as it only considers the overall performance of the model without taking into account class imbalance. 
- By prioritizing F1 over AUC-ROC, I aimed to ensure that my models were optimized for real-world applications.

# Results

The table below summarizes the F1 scores of the tested models on the six labels associated with toxic comments, along with their F1 macro average. In addition, the graphical visualization provided by the plot allows for an easier comparison between the models' performances across different labels.

<div align="center">
  
| Model Name         | toxic | severe_toxic | obscene | threat | insult | identity_hate | F1 macro avg |
|--------------------|-------|--------------|---------|--------|--------|---------------|--------------|
| Model I (Baseline) | 0.65  | 0.40         | 0.68    | 0.45   | 0.64   | 0.56          | 0.56         |
| Model II (Glove)   | 0.67  | 0.38         | 0.68    | 0.42   | 0.64   | 0.51          | 0.55         |
| Model III (BERT)   | 0.67  | 0.42         | 0.70    | 0.59   | 0.70   | 0.62          | 0.62         |
  
<img width="1948" height="784" alt="image" src="https://github.com/user-attachments/assets/680b702e-dd6f-4ce9-bd03-cee5283ebb71" />



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

<img width="952" height="358" alt="image" src="https://github.com/user-attachments/assets/de893363-7bf7-456b-adad-c8876a7da40d" />

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
![Screenshot 2025-08-04 151743](https://github.com/user-attachments/assets/8727e84d-7d13-4ddc-b48c-e6eb0d517b54)
![Screenshot 2025-08-04 151641](https://github.com/user-attachments/assets/a7eb52b9-1cef-4a3c-b7e8-cde8b4194fdb)


---

## Model Deployment
- BERT model served via **Gradio** on Hugging Face Spaces
- UI for entering comments and viewing toxicity predictions

<img width="928" height="347" alt="image" src="https://github.com/user-attachments/assets/096975b2-b90f-4397-a44c-0de860f12450" />


## Orchestration
- Prefect flows:
  - **Training**: runs model training and logs to MLflow
  - **Monitoring**: checks for drift and missing values
  - ![Screenshot 2025-08-04 185713](https://github.com/user-attachments/assets/5d9eaa8e-b281-4ddd-9714-e462cba65c4b)
    ![Screenshot 2025-08-04 185619](https://github.com/user-attachments/assets/5cecfcbc-6c63-49a5-aca2-641e72b9a9d7)



---

## Monitoring
- Drift metrics: `drift_score`, `missing_share`, `drifted_features`
- Stored in PostgreSQL and visualized in Grafana
- Alerts can be configured for threshold violations

![Screenshot 2025-08-05 005230](https://github.com/user-attachments/assets/3acd0708-019c-4fef-ae20-4e9fd6d73090)


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
