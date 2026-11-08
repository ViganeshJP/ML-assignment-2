import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Classification Model Evaluation", page_icon="🧪", layout="wide")

st.title("Machine Learning Classification Model Evaluation")
st.write(
    "This application evaluates the five pre-trained classification models "
    "on an uploaded test CSV. Models are loaded from disk; no retraining is performed."
)

MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def get_expected_features():
    from sklearn.datasets import load_breast_cancer
    return list(load_breast_cancer().feature_names)

expected_features = get_expected_features()

st.sidebar.header("Evaluation settings")
selected_model = st.sidebar.selectbox("Select classification model", list(MODEL_FILES.keys()))
uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"])

st.info(
    "The CSV must contain all 30 input features and a target column named `target`. "
    "Target labels: 0 = malignant, 1 = benign."
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        missing = [col for col in expected_features if col not in data.columns]
        if missing:
            st.error("The uploaded CSV is missing required feature columns: " + ", ".join(missing))
            st.stop()
        if "target" not in data.columns:
            st.error("The uploaded CSV must include the target column named `target`.")
            st.stop()
        if data.empty:
            st.error("The uploaded CSV contains no rows.")
            st.stop()

        X_eval = data[expected_features].apply(pd.to_numeric, errors="coerce")
        y_true = pd.to_numeric(data["target"], errors="coerce")

        if X_eval.isnull().any().any() or y_true.isnull().any():
            st.error("The CSV contains missing or non-numeric values in required columns.")
            st.stop()
        if not set(y_true.unique()).issubset({0, 1}):
            st.error("The target column must contain only 0 (malignant) and 1 (benign).")
            st.stop()

        model_path = os.path.join(MODEL_DIR, MODEL_FILES[selected_model])
        if not os.path.exists(model_path):
            st.error(f"Saved model not found: {model_path}. Ensure the model folder is in the repository.")
            st.stop()

        model = load_model(model_path)
        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval)[:, 1]

        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
            "Precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            "Recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred),
        }

        st.subheader(f"Results: {selected_model}")
        st.caption(f"Evaluated {len(data)} uploaded observations. Positive class for Precision, Recall and F1 is class 1 (benign).")
        cols = st.columns(6)
        for col, (label, value) in zip(cols, metrics.items()):
            col.metric(label, "N/A" if pd.isna(value) else f"{value:.4f}")

        left, right = st.columns(2)
        with left:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Malignant", "Benign"],
                yticklabels=["Malignant", "Benign"]
            )
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("Actual label")
            st.pyplot(fig)
            plt.close(fig)

        with right:
            st.subheader("Classification Report")
            report = classification_report(
                y_true, y_pred, labels=[0, 1],
                target_names=["Malignant", "Benign"],
                output_dict=True, zero_division=0
            )
            st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

        st.subheader("Prediction preview")
        preview = data.copy()
        preview["Predicted target"] = y_pred
        preview["Predicted diagnosis"] = pd.Series(y_pred).map({0: "Malignant", 1: "Benign"}).values
        st.dataframe(preview.head(20), use_container_width=True)

        st.download_button(
            "Download predictions CSV",
            preview.to_csv(index=False).encode("utf-8"),
            file_name="classification_predictions.csv",
            mime="text/csv"
        )
    except Exception as exc:
        st.error(f"Could not evaluate the uploaded file: {exc}")
else:
    st.write("Upload the provided `test_data.csv` file to view model evaluation results.")

st.divider()
st.caption("Academic demonstration using the Breast Cancer Wisconsin Diagnostic dataset. This application is not a medical diagnostic tool.")
