# 💳 Credit Card Fraud Detection System (Streamlit App)

An end-to-end **Machine Learning Web Application** built using **Streamlit** to detect fraudulent credit card transactions.

This project demonstrates a complete ML pipeline including data loading, preprocessing, model training, evaluation, explainability, and real-time prediction.

---

## 🚀 Project Overview

Credit card fraud is a significant financial issue worldwide.
This application helps in identifying fraudulent transactions using multiple machine learning models and provides explainability using SHAP.

The system allows users to:

* 📊 Upload and explore transaction data
* ⚙️ Train and compare multiple ML models
* 📈 Evaluate model performance using advanced metrics
* 🧠 Understand model predictions using SHAP
* 🔍 Predict fraud on new transactions
* 🔄 Run real-time fraud detection simulation
* 📥 Download trained models and evaluation reports

---

## 🧠 Machine Learning Models Used

The following models are implemented:

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM
* CatBoost
* 🤖 AutoML (Compare All Models)

### 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* ROC Curve

---

## 📊 Dataset Structure

The dataset must contain the following columns:

| Feature  | Description                                |
| -------- | ------------------------------------------ |
| Time     | Seconds elapsed since first transaction    |
| Amount   | Transaction amount                         |
| V1 - V28 | PCA anonymized features                    |
| Class    | Target variable (0 = Non-Fraud, 1 = Fraud) |

If no dataset is uploaded, the application generates a synthetic sample dataset.

---

## 🧩 Application Modules

### 📘 1. Overview

* Project introduction
* Dataset explanation
* Fraud detection objective

### 📊 2. EDA & Data Loader

* Upload CSV file
* View dataset preview
* Missing value check
* Class distribution visualization
* Histogram & Boxplots
* Correlation heatmap

### ⚙️ 3. Model Training

* Feature scaling using StandardScaler
* Train-test split
* Single model training
* AutoML comparison
* Leaderboard generation
* Confusion Matrix & ROC Curve
* Download trained model (.pkl)
* Download evaluation report (.csv)

### 🧠 4. Model Explainability

* SHAP Summary Plot
* SHAP Feature Importance (Bar Plot)
* SHAP Dependence Plot

### 🔍 5. Prediction

* Manual transaction input form
* Fraud probability prediction
* Real-time fraud detection simulation

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost
* SHAP
* Matplotlib
* Seaborn
* Plotly
* Joblib

---

## 📂 Project Structure

```
Credit-Card-Fraud-Detection/
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2️⃣ Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
venv\Scripts\activate   # For Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📥 Outputs

After training, you can:

* ✅ Download the trained model (.pkl file)
* 📄 Download evaluation report (.csv file)

---

## 🔮 Future Improvements

* Handle class imbalance using SMOTE
* Hyperparameter tuning
* Docker containerization
* Cloud deployment (Streamlit Cloud / AWS / Azure)
* Real-time API integration

---

## 👨‍💻 Author

**Adarsh Kumar**
Machine Learning & Data Science Enthusiast

---

## ⭐ If you found this project helpful, please give it a star!
