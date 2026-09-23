import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection")


st.sidebar.header("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    st.write(" Dataset Overview")
    st.write(data.head())
    st.write("Shape:", data.shape)
    st.write("Class distribution:")
    st.bar_chart(data['Class'].value_counts())

    
    scaler = StandardScaler()
    data['scaled_Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    data['scaled_Time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))
    data = data.drop(['Time', 'Amount'], axis=1)

    X = data.drop('Class', axis=1)
    y = data['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    st.write(f"Training set shape: {X_train.shape}")
    st.write(f"Test set shape: {X_test.shape}")

    
    st.subheader("Logistic Regression")
    model = LogisticRegression(class_weight='balanced', random_state=42, solver='liblinear')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    col1_lr, col2_lr = st.columns([1,3])
    with col1_lr:
        fig, ax = plt.subplots(figsize=(3,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'], ax=ax)
        st.pyplot(fig)

    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=['Not Fraud (0)', 'Fraud (1)']))

    auc_score = roc_auc_score(y_test, y_pred_proba)
    accuracy_lr = accuracy_score(y_test, y_pred)
    st.write(f"ROC-AUC Score (Logistic Regression): {auc_score:.4f}")
    st.write(f"Accuracy (Logistic Regression): {accuracy_lr:.4f}")

   
    st.subheader("Isolation Forest (Anomaly Detection)")
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    iso_forest.fit(X_train)
    y_pred_iso = iso_forest.predict(X_test)
    y_pred_iso = [0 if x==1 else 1 for x in y_pred_iso]

    cm_iso = confusion_matrix(y_test, y_pred_iso)
    st.write("Confusion Matrix:")
    col1_lro, col2_lr0 = st.columns([1,3])
    with col1_lro:
        fig2, ax2 = plt.subplots(figsize=(3,3))
        sns.heatmap(cm_iso, annot=True, fmt='d', cmap='Reds', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'], ax=ax2)
        st.pyplot(fig2)

    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred_iso, target_names=['Not Fraud (0)', 'Fraud (1)'])) 

   
    try:
        auc_score_iso = roc_auc_score(y_test, y_pred_iso)
    except Exception:
        auc_score_iso = None

    accuracy_iso = accuracy_score(y_test, y_pred_iso)
    st.write(f"ROC-AUC Score (Isolation Forest): {auc_score_iso:.4f}" if auc_score_iso is not None else "ROC-AUC Score (Isolation Forest): N/A")
    st.write(f"Accuracy (Isolation Forest): {accuracy_iso:.4f}")

    
    if accuracy_lr > accuracy_iso:
        best = "Logistic Regression"
        reason = f"higher accuracy ({accuracy_lr:.4f} vs {accuracy_iso:.4f})"
    elif accuracy_iso > accuracy_lr:
        best = "Isolation Forest"
        reason = f"higher accuracy ({accuracy_iso:.4f} vs {accuracy_lr:.4f})"
    else:
        
        if auc_score is not None and auc_score_iso is not None:
            if auc_score > auc_score_iso:
                best = "Logistic Regression"
                reason = f"tie on accuracy; better ROC-AUC ({auc_score:.4f} vs {auc_score_iso:.4f})"
            elif auc_score_iso > auc_score:
                best = "Isolation Forest"
                reason = f"tie on accuracy; better ROC-AUC ({auc_score_iso:.4f} vs {auc_score:.4f})"
            else:
                best = "Both (tie)"
                reason = f"same accuracy and ROC-AUC"
        else:
            best = "Both (tie)"
            reason = "same accuracy"

    st.markdown("### Best algorithm")
    st.success(f"{best} — {reason}")

else:
    st.info("Please upload a CSV file to continue.")