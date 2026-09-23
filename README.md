# 💳 Credit Card Fraud Detection Using Machine Learning

A Machine Learning-based web application for detecting fraudulent credit card transactions. The application is built using **Python** and **Streamlit** and compares two machine learning approaches:

* Logistic Regression
* Isolation Forest

The system allows users to upload a credit card transaction dataset, preprocesses the data, trains the models, evaluates their performance, and displays the results through an interactive Streamlit interface.

---

## 📌 Project Overview

Credit card fraud is a major problem in digital financial transactions. Fraudulent transactions usually represent only a very small percentage of the total transactions, making fraud detection an **imbalanced classification problem**.

This project uses machine learning to identify potentially fraudulent transactions and compares a supervised classification algorithm with an anomaly detection algorithm.

---

## 🎯 Objectives

* Detect fraudulent credit card transactions using Machine Learning.
* Preprocess transaction data before model training.
* Handle imbalanced data using class weighting.
* Compare Logistic Regression and Isolation Forest.
* Evaluate models using different performance metrics.
* Visualize results using confusion matrices.
* Provide an easy-to-use web interface using Streamlit.

---

## 🤖 Machine Learning Algorithms

### 1. Logistic Regression

Logistic Regression is a supervised machine learning algorithm used for binary classification.

In this project:

* `Class = 0` represents a normal transaction.
* `Class = 1` represents a fraudulent transaction.
* `class_weight='balanced'` is used to reduce the effect of class imbalance.

### 2. Isolation Forest

Isolation Forest is an unsupervised anomaly detection algorithm.

It detects unusual transactions by isolating abnormal data points from normal transactions.

In this project:

* Normal transaction → `0`
* Fraudulent/anomalous transaction → `1`

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn

---

## 📂 Dataset Requirements

The application accepts a **CSV file**.

The dataset should contain:

* `Time`
* `Amount`
* `Class`
* Other numerical transaction features

The target column should be:

```text
Class
```

where:

```text
0 = Normal Transaction
1 = Fraudulent Transaction
```

---

## ⚙️ Project Workflow

```text
Upload CSV Dataset
        ↓
Dataset Exploration
        ↓
Data Preprocessing
        ↓
Standard Scaling
        ↓
Train-Test Split
        ↓
 ┌──────────────────────┐
 │                      │
 ↓                      ↓
Logistic Regression   Isolation Forest
 │                      │
 ↓                      ↓
Predictions           Anomaly Detection
 │                      │
 └──────────┬───────────┘
            ↓
     Model Evaluation
            ↓
 Confusion Matrix
 Classification Report
 Accuracy & ROC-AUC
            ↓
   Best Model Comparison
```

---

## 🔄 Data Preprocessing

The `Time` and `Amount` columns are standardized using:

```python
StandardScaler()
```

New features are created:

```text
scaled_Time
scaled_Amount
```

The original `Time` and `Amount` columns are then removed.

The dataset is divided into:

```text
80% → Training Data
20% → Testing Data
```

Stratified sampling is used to preserve the fraud/non-fraud class distribution.

---

## 📊 Model Evaluation

The models are evaluated using:

### Confusion Matrix

The confusion matrix shows:

* True Positive
* True Negative
* False Positive
* False Negative

### Classification Report

The report contains:

* Precision
* Recall
* F1-Score
* Support

### Accuracy

Accuracy measures the percentage of correctly classified transactions.

### ROC-AUC Score

ROC-AUC measures how effectively the model separates fraudulent transactions from legitimate transactions.

---

## 🏆 Model Comparison

After training both models, the application compares their accuracy.

If both models have the same accuracy, the ROC-AUC score is used as an additional comparison.

The application then displays the algorithm with the better result.

> **Note:** In highly imbalanced fraud-detection datasets, accuracy alone can be misleading. Precision, recall, F1-score, ROC-AUC, and the confusion matrix should also be considered when evaluating the models.

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-project-folder>
```

### 2. Install Required Libraries

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn
```

Alternatively, create a `requirements.txt` file containing:

```text
streamlit
pandas
scikit-learn
matplotlib
seaborn
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

If your main Python file is named `app.py`, run:

```bash
streamlit run app.py
```

Streamlit will start the application and provide a local URL such as:

```text
http://localhost:8501
```

Open it in your browser.

---

## 💻 How to Use

1. Start the Streamlit application.
2. Open the application in your browser.
3. Click **Upload CSV file** in the sidebar.
4. Select the credit card transaction dataset.
5. View the dataset overview and class distribution.
6. Wait for the models to train.
7. View the Logistic Regression results.
8. View the Isolation Forest results.
9. Compare the confusion matrices, classification reports, accuracy, and ROC-AUC scores.
10. View the model comparison result.

---

## 📁 Suggested Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── app.py
├── requirements.txt
├── README.md
└── dataset/
    └── creditcard.csv
```

---

## ✨ Features

* Interactive Streamlit web application
* CSV dataset upload
* Automatic data preprocessing
* Feature scaling
* Train-test splitting
* Logistic Regression fraud detection
* Isolation Forest anomaly detection
* Confusion matrix visualization
* Classification report
* Accuracy calculation
* ROC-AUC evaluation
* Automatic model comparison

---

## 🔮 Future Improvements

The project can be extended by adding:

* Random Forest
* XGBoost
* Decision Tree
* Neural Networks
* SMOTE for handling class imbalance
* Precision-Recall curves
* ROC curve visualization
* Feature importance analysis
* Hyperparameter tuning
* Real-time transaction prediction
* Database integration
* Deployment on cloud platforms

---

## 📌 Conclusion

This project demonstrates how Machine Learning can be used to detect fraudulent credit card transactions.

**Logistic Regression** provides a supervised classification approach, while **Isolation Forest** detects transactions that behave like anomalies. The Streamlit interface makes it easy to upload transaction data, train both models, visualize their performance, and compare the results.

---

## 👨‍💻 Author

**Balamurugan U**

B.E. Computer Science and Engineering
Artificial Intelligence & Machine Learning
Jerusalem College of Engineering

---

⭐ If you find this project useful, consider giving the repository a star.
