# 💳 AI-Powered Credit Card Fraud Detection and Risk Analytics

An end-to-end Machine Learning application that detects fraudulent credit card transactions using multiple classification models and provides an interactive analytics dashboard built with Streamlit.

---

## 📌 Project Overview

Financial fraud is one of the biggest challenges faced by banks and payment gateways. This project leverages Machine Learning algorithms to identify potentially fraudulent credit card transactions by analyzing historical transaction patterns.

The application predicts whether a transaction is fraudulent, assigns a fraud risk score (0–100), categorizes transactions into different risk levels, and presents the results through an interactive dashboard.

---

## 🚀 Features

* Exploratory Data Analysis (EDA)
* Data Preprocessing
* Duplicate Removal
* Feature Scaling
* Handling Class Imbalance
* Logistic Regression Model
* Random Forest Model
* XGBoost Model
* Hyperparameter Optimization
* Model Comparison
* Confusion Matrix
* ROC Curve
* Fraud Risk Score (0–100)
* Risk Level Classification
* Interactive Streamlit Dashboard
* CSV Upload for Batch Predictions
* Download Prediction Results
* Analytics Dashboard

---

## 📂 Dataset

This project uses the **Credit Card Fraud Detection** dataset available on Kaggle.

**Dataset Link**

[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

> Due to GitHub's file size limit, the dataset is not included in this repository. Download `creditcard.csv` from Kaggle and place it in the project root before running the notebook.

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* XGBoost
* Imbalanced-Learn (SMOTE)
* Joblib
* Streamlit
* Git & GitHub

---

## 📁 Project Structure

```text
fraud_detection/
│
├── app.py
├── fraud_detection.ipynb
├── fraud_model.pkl
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Aaryansrinivas/fraud_detection.git
```

Move into the project

```bash
cd fraud_detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Download the Kaggle dataset and place

```
creditcard.csv
```

inside the project folder.

---

## ▶️ Run the Notebook

Open

```
fraud_detection.ipynb
```

and execute all cells.

---

## ▶️ Run the Streamlit App

```bash
streamlit run app.py
```

Open your browser at

```
http://localhost:8501
```

---

## 📊 Dashboard Features

* Total Transactions
* Fraud Detected Count
* Genuine Transactions
* Average Risk Score
* Fraud Distribution
* Risk Level Distribution
* Top Suspicious Transactions
* High-Risk Transactions
* CSV Export

---

## 🤖 Machine Learning Models

* Logistic Regression
* Random Forest
* XGBoost

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

## 📈 Fraud Risk Score

Each transaction is assigned a risk score between **0 and 100**.

| Risk Score | Risk Level |
| ---------- | ---------- |
| 0 – 29     | Low        |
| 30 – 69    | Medium     |
| 70 – 100   | High       |

---
---

## 🎯 Future Improvements

* SHAP Explainability
* FastAPI Backend
* Docker Support
* Real-Time Prediction API
* Database Integration
* User Authentication
* Model Monitoring
* Cloud Deployment

---

## 📚 Learning Outcomes

Through this project, I gained practical experience in:

* Data Cleaning
* Feature Engineering
* Handling Imbalanced Data
* Machine Learning Model Development
* Hyperparameter Optimization
* Model Evaluation
* Model Serialization
* Interactive Dashboard Development
* Deployment using Streamlit
* Version Control with Git & GitHub

---

## 👨‍💻 Author

**Aaryan Srinivas**



---

## ⭐ If you found this project useful, consider giving it a star!
