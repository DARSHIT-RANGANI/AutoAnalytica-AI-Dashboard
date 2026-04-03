# 🚀 AutoAnalytica-AI — Smart AutoML with RL Intelligence

## 📌 Overview

**AutoAnalytica-AI** is a next-generation **AI-powered AutoML platform** that acts like a **self-learning data scientist**.

It automates the complete machine learning lifecycle — from **data preprocessing to model selection, optimization, and explainability** — enhanced with **Reinforcement Learning (RL)** and **Meta-Learning**.

This project is designed for **real-world ML automation**, **research**, and **portfolio showcase**.

---

## 📷 Application Preview

<p align="center">
  <img src="YOUR_SCREENSHOT_1" ><img width="263" height="2048" alt="screencapture-localhost-3000-2026-03-28-12_36_26" src="https://github.com/user-attachments/assets/72f7c782-f41c-4d60-a70a-158cb581f60d"/>
</p>

<br>

<p align="center">
  <img width="263" height="2048" alt="screencapture-localhost-3000-2026-03-28-12_36_49" src="https://github.com/user-attachments/assets/02c8149a-1eea-43c2-bab5-7d92240bc612"/>
</p>

<br>

<p align="center">
  <img src="YOUR_SCREENSHOT_3" ><img width="263" height="2048" alt="fullpage" src="https://github.com/user-attachments/assets/66b69e79-dcaf-46f8-9f97-93fb56d5ae52"/>
</p>

<br>

---

## ✨ Key Features

| 🚀 Feature           | 💡 Description                              |
| :------------------- | :------------------------------------------ |
| 📂 Dataset Upload    | Upload CSV datasets with validation         |
| 🧹 Data Cleaning     | Automatic universal preprocessing           |
| 📊 Diagnostics       | Dataset analysis + problem type detection   |
| 🧠 Meta Learning     | Ranks best ML models intelligently          |
| 🤖 AutoML            | Trains multiple ML algorithms               |
| 🔁 RL Bandit         | Dynamic model selection during training     |
| 🧪 Stability Testing | Multi-seed validation (42, 0, 99)           |
| 🧠 Ensemble Learning | Automatic stacking models                   |
| 📑 Reports           | Performance insights & downloadable reports |
| 🔍 Explainability    | SHAP-based feature importance               |

---

## 🧠 Real AI Pipeline (Actual System Flow)

```text
Upload Data
   ↓
load_data()
   ↓
universal_cleaning()
   ↓
detect_problem_type()
   ↓
log_dataset_diagnostics()
   ↓
detect_data_leakage()
   ↓
auto_feature_selection()  [multi-step]
   ↓
MetaPriorityScorer (meta-learning ranking)
   ↓
RL Bandit Loop (ε-greedy model selection)
   ↓
Hyperparameter Tuning (GridSearch / RandomSearch)
   ↓
Multi-seed Stability (seeds: 42, 0, 99)
   ↓
Stacking Ensemble (top models)
   ↓
detect_overfitting()
   ↓
calculate_confidence()
   ↓
_get_shap_explanation()
   ↓
save_model() + generate_report()
```

---

## ⚠️ Important Clarifications

| ❌ Misconception               | ✅ Reality                        |
| :---------------------------- | :------------------------------- |
| Smart Decision Engine exists  | Logic is distributed (Meta + RL) |
| RL runs after training        | RL runs **during training**      |
| Deep Learning models included | Only classical ML used           |
| AI explanation engine         | Uses SHAP internally             |
| EDA is part of pipeline       | Separate module (not used here)  |

---

## 🧠 Core Intelligence Components

### 🔹 MetaPriorityScorer

* Learns from previous runs
* Ranks candidate models
* Speeds up AutoML decisions

### 🔹 RL Bandit (System Brain)

* Uses **ε-greedy strategy**
* Selects best model dynamically
* Learns from CV scores

### 🔹 Feature Selection

* Multi-step filtering pipeline
* Improves accuracy & efficiency

### 🔹 Stacking Ensemble

* Combines top-performing models
* Produces stronger predictions

### 🔹 Confidence Scoring

* Measures reliability of predictions

### 🔹 SHAP Explainability

* Explains model decisions clearly

---

## 🛠️ Tech Stack

### ⚛️ Frontend

* React.js
* Tailwind CSS
* Axios

### ⚡ Backend

* FastAPI
* Python 3.13
* Pydantic

### 🤖 Machine Learning

* Scikit-learn
* XGBoost
* LightGBM
* CatBoost

### 🗄️ Database

* MongoDB
* MongoDB Compass

---

## 📁 Project Structure

```text
autoanalytica-ai/
├── ai-frontend/
│   ├── public/
│   ├── src/
│   │   ├── features/
│   │   ├── components/
│   │   ├── services/
│   │   └── utils/
│
├── backend/
│   ├── app/
│   │   ├── api/v1/
│   │   ├── core/
│   │   ├── db/
│   │   ├── services/ml/
│   │   ├── services/agents/
│   │   └── schemas/
│   │
│   ├── tests/
│   ├── data/
│   └── requirements.txt
│
├── docs/
├── scripts/
└── README.md
```

---

## 🔌 API Endpoints

### 📤 Dataset APIs

```text
POST   /upload/api/upload
GET    /upload/api/datasets
GET    /upload/api/datasets/{id}
DELETE /upload/api/datasets/{id}
```

### 🤖 Model APIs

```text
POST   /ai/train
GET    /ai/models
GET    /ai/models/{id}
DELETE /ai/models/{id}
```

---

## 🗄️ Database Setup (MongoDB)

### ▶️ Start MongoDB

```bash
net start MongoDB
```

### ▶️ Connect via MongoDB Compass

```bash
mongodb://localhost:27017
```

### 📂 Database: `autoanalytica_db`

Collections:

* datasets
* models
* experiences
* users

---

## 🚀 Getting Started

### 🔧 Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

### 💻 Frontend Setup

```bash
cd ai-frontend
npm install
npm start
```

---

## 🧪 Testing

```bash
cd backend
pytest
```

---

## 🎯 What Makes This Project Unique?

✔ RL integrated inside training loop
✔ Meta-learning driven model ranking
✔ Fully automated ML pipeline
✔ Multi-seed stability validation
✔ Automatic stacking ensemble
✔ Explainable AI using SHAP

---

## 🎓 Ideal For

* Data Science Students
* ML Engineers
* Full Stack Developers
* AI Researchers
* Portfolio Projects

---

## 🔮 Future Scope

* Deep Learning (ANN, TabNet)
* LLM-based explanations
* Cloud deployment (AWS / GCP)
* Docker & CI/CD
* Real-time prediction APIs

---

## 👨‍💻 Author

**Darshit Rangani**
B.Tech Computer Engineering (Data Science)

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
