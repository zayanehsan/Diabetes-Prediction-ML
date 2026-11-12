# Diabetes Prediction Using Machine Learning

## 1. Project Overview

This project uses machine learning techniques to predict diabetes based on patient health indicators. It uses the CDC BRFSS 2015 Diabetes Health Indicators dataset to identify patterns associated with diabetes.

The project includes data exploration, preprocessing, model training, hyperparameter tuning, model evaluation, and a patient risk assessment module with general health recommendations.

## 2. Problem Statement

Diabetes is a major health concern worldwide. Early identification of potential diabetes risk can encourage individuals to seek medical testing and professional advice.

The objective of this project is to develop a machine learning model that analyzes health-related information and estimates whether a person may have diabetes.

The system also provides general health precautions when the model indicates potential diabetes risk.

## 3. Project Objectives

- Analyze health indicators associated with diabetes.
- Perform exploratory data analysis (EDA).
- Clean and preprocess the dataset.
- Handle class imbalance using Random Oversampling.
- Train and evaluate multiple machine learning models.
- Apply hyperparameter tuning using GridSearchCV.
- Compare model performance using appropriate evaluation metrics.
- Identify important features influencing predictions.
- Develop a patient risk assessment and health recommendation module.

## 4. Dataset Information

**Dataset:** CDC BRFSS 2015 Diabetes Health Indicators

**Dataset file:** `diabetes_012_health_indicators_BRFSS2015.csv`

**Dataset source:** [Kaggle – Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

| Property | Description |
|---|---|
| Original records | 253,680 |
| Total columns | 22 |
| Independent features | 21 |
| Target variable | Diabetes_012 |
| Missing values | 0 |

### Target Variable

The original target variable contains three categories:

- 0: No diabetes
- 1: Prediabetes
- 2: Diabetes

For this project, the task was converted into binary classification:

- 0: No diabetes
- 1: Diabetes

The prediabetes category was excluded from the binary classification task.

After removing exact duplicate records, the prepared dataset contained **225,152 records**.

## 5. Technologies and Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- XGBoost
- Matplotlib
- Seaborn
- Joblib

## 6. Project Structure

```text
Diabetes_ML_Project/
│
├── Code/
│   ├── EDA.py
│   ├── Preprocessing.py
│   ├── Health_Recommendations.py
│   │
│   ├── Models/
│   │   ├── Baseline.py
│   │   ├── Logistic_Regression.py
│   │   ├── Decision_Tree.py
│   │   ├── Random_Forest.py
│   │   ├── Naive_Bayes.py
│   │   ├── SVM.py
│   │   ├── Gradient_Boosting.py
│   │   └── XGBoost.py
│   │
│   ├── Tuned_Models/
│   │
│   └── Result_Scripts/
│
├── Dataset/
│   └── diabetes_012_health_indicators_BRFSS2015.csv
│
├── Processed_Data/
│   ├── X_train.csv
│   ├── Y_train.csv
│   ├── X_test.csv
│   ├── Y_test.csv
│   ├── X_train_original.csv
│   ├── Y_train_original.csv
│   ├── X_train_raw.csv
│   └── X_test_raw.csv
│
├── Graphs/
│   ├── EDA/
│   ├── Model_Comparison/
│   ├── ROC_PR/
│   ├── Confusion_Matrices/
│   └── Feature_Importance/
│
├── Results/
│   ├── error_analysis.csv
│   ├── feature_importance.csv
│   ├── final_diabetes_model.pkl
│   ├── final_model_comparison.csv
│   ├── model_metrics.csv
│   ├── ROC_PR_Metrics.csv
│   ├── threshold_analysis.csv
│   ├── threshold_test_results.csv
│   └── tuning_results.csv
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 7. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset and identify patterns in the health indicators.

The analysis included:

- Dataset dimensions and data types.
- Missing value analysis.
- Duplicate record analysis.
- Target class distribution.
- Descriptive statistics.
- Distribution of numerical features.
- Outlier detection using the IQR method.
- Correlation analysis.
- Relationships between health indicators and diabetes.

Visualizations were generated to identify patterns associated with diabetes, including BMI, age, general health, physical activity, high blood pressure, and cholesterol.

## 8. Data Preprocessing

The following preprocessing steps were performed:

1. Removed the prediabetes category to create a binary classification problem.
2. Removed exact duplicate records.
3. Separated independent features and the target variable.
4. Applied stratified 70/30 training and testing split using random state 42.
5. Applied StandardScaler to numerical features.
6. Used RandomOverSampler to address class imbalance in training data.
7. Kept the testing dataset unchanged and free from oversampling.

### Dataset Split

| Dataset | Records |
|---|---:|
| Training data before oversampling | 157,606 |
| Testing data | 67,546 |
| Training data after oversampling | 180,950 |

The oversampled training data contains 133,038 majority-class records and 47,912 minority-class records.

For hyperparameter tuning, preprocessing and oversampling were performed within an imbalanced-learn pipeline to avoid data leakage during cross-validation.

## 9. Machine Learning Models

The project implemented eight models, including a baseline classifier.

1. Dummy Classifier (Baseline)
2. Logistic Regression
3. Decision Tree
4. Random Forest
5. Gaussian Naive Bayes
6. Support Vector Machine (Linear SVM)
7. Gradient Boosting
8. XGBoost

The baseline model was used to establish a reference performance against which the machine learning models were compared.

## 10. Hyperparameter Tuning

Hyperparameter tuning was performed on seven machine learning models using GridSearchCV.

The tuning process included:

- Five-fold Stratified Cross-Validation.
- GridSearchCV for parameter optimization.
- F1-score as the primary optimization metric.
- Random oversampling within the training pipeline.
- Evaluation on the untouched test dataset.

The selected parameters were determined through cross-validation.

## 11. Model Evaluation

Models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1-score
- ROC-AUC
- Average Precision (PR-related evaluation)
- Confusion Matrix

Additional evaluation included:

- Model comparison graphs.
- ROC curves.
- Precision-Recall curves.
- Confusion matrix visualizations.
- Feature importance analysis.
- Error analysis.
- Classification threshold analysis.

Recall was given particular consideration because identifying potential diabetes cases is important in a health-related prediction task.

## 12. Final Model

The tuned Gradient Boosting classifier was selected as the final model based on its cross-validation F1-score and overall performance.

### Selected Hyperparameters

| Parameter | Value |
|---|---|
| Model | Gradient Boosting |
| Number of estimators | 150 |
| Learning rate | 0.1 |
| Maximum depth | 3 |
| Classification threshold | 0.375 |

The selected threshold of 0.375 was determined through five-fold out-of-fold threshold analysis on training data.

### Final Test Results

At the selected threshold of 0.375:

| Metric | Result |
|---|---:|
| Accuracy | 79.52% |
| Precision | 39.92% |
| Recall | 62.23% |
| F1-score | 48.64% |
| Specificity | 82.71% |

The selected threshold improves recall compared with the default 0.5 threshold, allowing the model to identify more positive cases at the cost of additional false positives.

The final trained model and its threshold are saved in:

`Results/final_diabetes_model.pkl`

## 13. Health Risk Assessment and Recommendations

The project includes an additional module named `Health_Recommendations.py`.

This module allows a user to enter 21 health-related features and receive a prediction using the saved Gradient Boosting model.

Its functionality includes:

- Collecting patient health information.
- Estimating diabetes probability.
- Applying the selected classification threshold of 0.375.
- Displaying the model's prediction.
- Providing general health precautions when potential diabetes risk is indicated.

General recommendations include maintaining a balanced diet, regular physical activity, healthy weight management, avoiding smoking, and consulting a healthcare professional.

This feature is educational and does not provide medical diagnosis or treatment.

## 14. How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/zayanehsan/Diabetes-Prediction-ML.git
```

### Step 2: Open the Project Folder

```bash
cd Diabetes-Prediction-ML
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Environment

Windows:

```bash
.venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run Exploratory Data Analysis

```bash
python Code/EDA.py
```

### Step 7: Run Data Preprocessing

```bash
python Code/Preprocessing.py
```

### Step 8: Run the Final Model

```bash
python Code/Models/Gradient_Boosting.py
```

### Step 9: Run Patient Risk Assessment

After the final model artifact has been generated:

```bash
python Code/Health_Recommendations.py
```

The patient assessment module requires the saved model in the `Results` folder.

## 15. Limitations

- The dataset is based on self-reported health information.
- The dataset represents BRFSS 2015 and may not reflect current population patterns.
- Prediabetes cases were excluded from the binary classification task.
- Identical feature combinations can have conflicting target labels.
- The model may not generalize equally well to every population.
- The model has not been clinically validated.
- Predictions should not be treated as confirmed medical diagnoses.

## 16. Future Improvements

- Develop a more user-friendly patient interface.
- Evaluate the model on newer health datasets.
- Explore additional feature selection techniques.
- Investigate explainable AI methods such as SHAP.
- Improve prediction performance while maintaining high recall.
- Conduct further validation using independent datasets.

## 17. Disclaimer

This project is developed for academic and educational purposes only.

The predictions generated by the machine learning model are estimates based on patterns in the dataset. They are not medical diagnoses and must not be used as a substitute for professional medical advice, testing, or treatment.

Individuals concerned about diabetes should consult qualified healthcare professionals.

## 18. Author

**Computer Science Student**  
University of Lahore

GitHub: [zayanehsan](https://github.com/zayanehsan)

---