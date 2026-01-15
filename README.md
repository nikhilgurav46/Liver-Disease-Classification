# 🩺 Liver Disease Classification using Machine Learning

This project focuses on building a machine learning system to classify liver disease conditions using clinical laboratory data. The model is trained with proper preprocessing, class imbalance handling, and is deployed as an interactive Streamlit web application.

---

## 📌 Problem Statement
Liver diseases such as hepatitis, fibrosis, and cirrhosis are difficult to detect early. This project aims to assist in early diagnosis by predicting liver disease categories based on patient test results.

---

## 📊 Dataset
- Medical liver disease dataset
- Numerical biochemical test features
- Categorical target variable (`category`)
- Highly imbalanced multi-class data

---

## ⚙️ Project Workflow

1. **Data Cleaning**
   - Missing value handling (median imputation)
   - Data type corrections

2. **Exploratory Data Analysis (EDA)**
   - Statistical summary
   - Histograms & KDE plots
   - Count plots for categorical data
   - Outlier detection using boxplots

3. **Feature Engineering**
   - Label Encoding for target variable
   - Feature Scaling using StandardScaler

4. **Handling Class Imbalance**
   - SMOTE (Synthetic Minority Over-sampling Technique)
   - Applied only on training data

5. **Model Training**
   - Logistic Regression
   - Decision Tree
   - Random Forest (Final Model)
   - XGBoost
   - Cost-sensitive learning using `class_weight`

6. **Model Evaluation**
   - Precision, Recall, F1-score
   - Classification Report
   - Focus on minority class performance

7. **Deployment**
   - Streamlit web application
   - Model, scaler, and label encoder saved using pickle

---

## 🧠 Final Model
- **Random Forest with SMOTE**
- Chosen based on better balance between precision and recall for minority disease classes

---

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost
- Streamlit

---

## Model Deployed on Render
Link: https://liver-disease-classification-vy6p.onrender.com

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
````

### 2️⃣ Train the Model

Run the Jupyter Notebook:

```bash
Predict Liver Disease.ipynb
```

### 3️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---


## 📌 Note

* SMOTE is applied **only during training**, not during prediction.
* Class imbalance is a key challenge in medical datasets and is handled carefully.
* The Final Model Selected was **Gradient Boost**

---

## 👤 Author

**Nikhil Gurav**
Aspiring Data Scientist | Machine Learning Enthusiast
