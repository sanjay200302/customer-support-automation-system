# Customer Support Automation System

## 📌 Overview
This project implements a customer support automation system that:
- Classifies customer messages into intents using **DistilBERT**.
- Decides whether to **auto‑handle or escalate** queries.
- Generates polite, brand‑aligned replies.
- Provides evaluation metrics, visualizations, and a decision log.

The system integrates **Databricks workflows** for preprocessing and storage with a **VS Code ML pipeline** for inference and reply generation.

---

## ⚙️ Setup Instructions

## 1. Clone the repository
```bash
git clone <repo-link>
cd <Project_root>
```

## 2. 📂Project structure

```text
Hiver-SDE-Intern-Project/
├── 1data/
│   ├── processed/
│   │   ├── AmazonHelp.csv
│   │   └── amazon_tweets.csv
│   └── raw/
│       ├── sample.csv
│       └── twcs.csv
├── configs/
│   ├── brand_config.yaml
│   └── model_config.yaml
├── notebooks/
│   ├── data/
│   │   ├── golden_set_balanced.csv
│   │   └── golden_set_labeled.csv
│   ├── models/intents/
│   ├── dataset_integration.ipynb
│   ├── escalation_test.ipynb
│   ├── evaluation.ipynb
│   ├── intent_classification.ipynb
│   ├── reply_generation.ipynb
│   └── databricks_preprocessing.dbc
├── report/
│   ├── decision_log.md
│   └── README.md
├── ResultsPics/
│   ├── 
│   └──
├── src/
│   ├── escalation.py
│   ├── evaluation.py
│   ├── intents.py
│   ├── merge_results.py
│   ├── reply_agent.py
│   ├── spark_preprocess.py
│   └── visualize_results.py
├── tests/
│   ├── test_escalation.py
│   ├── test_evaluation.py
│   ├── test_intents.py
│   └── test_reply_agent.py
├── baseline_pipeline.py
├── README.md
├── requirements.txt
└── run_pipeline.py
```

## 3. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Pipeline

### Baseline pipeline

```bash
python baseline_pipeline.py
```

### Full ML pipeline

```bash
python databricks_full_pipeline.py
```

### Generate visualizations

```bash
python visualize_results.py
```

### 🔮 Future Work
---

Deploy pipeline on Cloud Run for real‑time serving.

Add monitoring dashboards for escalation trends.

Improve reply diversity with RLHF (Reinforcement Learning from Human Feedback).

Explore hierarchical intent classification to reduce overlap between similar categories.

---

### 👥 Contributor

SanjayKumar – AIML Engineer, project lead and developer.
