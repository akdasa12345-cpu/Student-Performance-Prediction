# Student Performance Prediction and Academic Risk Analysis
## 1. Introduction
Academic performance is influenced by attendance, study habits, previous achievement, assignment work, and internal assessment. This project uses data science and machine learning to explore these factors and support early identification of students who may need academic assistance.

## 2. Objectives
- Analyse student attendance, study hours, previous marks, assignment marks, and internal marks.
- Predict final examination marks using Linear Regression or an ensemble regression model.
- Predict Pass/Fail status using Logistic Regression.
- Use KNN to identify students with similar academic patterns.
- Use K-Means to explore High Performing, Average, and Academically At-Risk groups.
- Evaluate models using a held-out test set and cross-validation.

## 3. Dataset and features
The included CSV is synthetic demonstration data. It contains attendance percentage, study hours per week, previous marks, assignment marks, internal marks, internet access, and final examination marks. No real student personal information is included.

## 4. Methodology
1. Load and inspect the dataset with Pandas.
2. Check missing values, duplicates, distributions, and correlations.
3. Impute missing numerical and categorical values.
4. Scale numerical features and one-hot encode categorical features.
5. Split data into training and test sets.
6. Train regression and classification models.
7. Evaluate predictions with MAE, R², accuracy, and a classification report.
8. Use KNN and K-Means for similarity and exploratory segmentation.

## 5. Models
- Linear Regression: baseline final-mark prediction (can be added for comparison).
- Random Forest Regressor: included as a nonlinear marks predictor.
- Logistic Regression: Pass/Fail classification.
- K-Nearest Neighbours: similar-pattern classification.
- K-Means: unsupervised student grouping.

## 6. Expected outputs
Predicted final marks, Pass/Fail probability/class, academic risk level (Low/Medium/High), and exploratory student segments.

## 7. Limitations and ethical use
Predictions are not decisions about a student's ability or future. They must not be used to punish or deny opportunities. Use representative, consented, anonymized data and review predictions with educators. Synthetic demonstration results must not be presented as findings from real students.

## 8. Conclusion
The project demonstrates a machine-learning workflow for academic performance analysis. Further work can include a larger approved dataset, fairness checks, explainability, and a simple web interface.
