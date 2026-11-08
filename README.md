# Machine Learning Assignment – 2
## Classification Models and Performance Evaluation

**Programme:** M.Tech (AIML)  
**Subject:** Machine Learning

## 1. Problem statement

The objective is to implement five classification algorithms on one public classification dataset, evaluate them using six performance metrics, and make the trained models available through an interactive Streamlit application.

This project classifies observations in the Breast Cancer Wisconsin Diagnostic dataset into malignant (0) and benign (1) classes.

## 2. Dataset description

- **Dataset:** Breast Cancer Wisconsin Diagnostic dataset
- **Source:** Scikit-learn built-in dataset loader (`sklearn.datasets.load_breast_cancer`)
- **Instances:** 569
- **Input features:** 30 numerical features
- **Target:** Binary classification (`0 = malignant`, `1 = benign`)
- **Class distribution:** 212 malignant and 357 benign observations
- **Missing values:** None in the loaded dataset
- **Split:** 80% training and 20% testing, stratified, with `random_state=42`
- **Test set:** 114 observations

The same training and test split was used for all five classifiers. StandardScaler is applied inside pipelines for Logistic Regression, KNN, Gaussian Naive Bayes and SVM. The Decision Tree is not scaled.

## 3. GitHub Repository Link

**Add your repository URL here after creating and uploading the project:**  
`https://github.com/<your-github-username>/ML-Assignment-2`

## 4. Models used and evaluation metrics

The positive class for Precision, Recall and F1-score is class 1 (benign). AUC is calculated from the predicted probability for class 1.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved 98.25% accuracy, 0.9954 AUC and 0.9861 F1-score. It has the highest AUC and ties for the highest accuracy, precision, recall and F1-score. |
| Decision Tree | Achieved 91.23% accuracy and 0.9286 F1-score. It has the lowest accuracy, AUC, recall, F1-score and MCC among the five models in this experiment. |
| KNN | Achieved 95.61% accuracy and 0.9655 F1-score, showing strong performance, although it is below Logistic Regression and SVM on the main classification metrics. |
| Naive Bayes | Achieved 92.98% accuracy and 0.9868 AUC. Its AUC is high, but its accuracy and F1-score are lower than those of Logistic Regression, SVM and KNN. |
| Support Vector Machine | Achieved 98.25% accuracy and 0.9950 AUC. It ties Logistic Regression on accuracy, precision, recall and F1-score, with a slightly lower AUC. |

### Overall winner

**Logistic Regression** is selected as the overall winner based on its highest AUC (0.9954). It also ties with SVM for the highest accuracy (0.9825), precision (0.9861), recall (0.9861), F1-score (0.9861) and MCC (0.9623). SVM is a very close competitor.

Because this is a medical diagnostic dataset, the malignant-class recall and false-negative count should also be reviewed before interpreting the model for any real-world clinical use. This project is for academic demonstration only and is not a medical diagnostic tool.

## 5. Streamlit application

The application (`app.py`) provides:
- CSV upload for held-out test data.
- Dropdown selection among all five saved classifiers.
- Calculation and display of Accuracy, AUC, Precision, Recall, F1-score and MCC.
- Confusion matrix and classification report.
- Preview and download of predictions.

The application loads pre-trained model files from the `model/` folder. It does not retrain the models on the uploaded test data. The uploaded CSV must contain all 30 input features and the `target` column.

**Live Streamlit App Link:** Add the deployed URL here after deployment.

## 6. Project files

```text
ML_Assignment_2/
├── app.py
├── requirements.txt
├── README.md
├── ML_Assignment_2.ipynb
├── test_data.csv
├── model_comparison.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib
```

## 7. How to run locally

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the Streamlit application from the project root:

   ```bash
   streamlit run app.py
   ```

3. Upload `test_data.csv`, select a model and review the displayed metrics.

## 8. Reproducibility

The saved models were recreated using the same dataset loader, train-test split, random state, preprocessing pipelines and model parameters documented in the notebook. The regenerated metric values were checked against the supplied `model_comparison.csv`.

## 9. BITS Virtual Lab execution proof

Insert the required screenshot of the notebook execution on BITS Virtual Lab in the final assignment PDF.
