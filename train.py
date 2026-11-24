from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "student_performance_demo.csv"
OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
target_marks = "final_exam_marks"
features = ["attendance_pct", "study_hours_per_week", "previous_marks",
            "assignment_marks", "internal_marks", "internet_access"]
X = df[features]
y_marks = df[target_marks]
y_pass = (y_marks >= 40).astype(int)

numeric = ["attendance_pct", "study_hours_per_week", "previous_marks",
           "assignment_marks", "internal_marks"]
categorical = ["internet_access"]
preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
])

X_train, X_test, ym_train, ym_test = train_test_split(
    X, y_marks, test_size=0.2, random_state=42
)
reg = Pipeline([("prep", preprocess), ("model", RandomForestRegressor(
    n_estimators=120, random_state=42))])
reg.fit(X_train, ym_train)
pred_marks = reg.predict(X_test)
joblib.dump(reg, OUT / "final_marks_model.joblib")

Xp_train, Xp_test, yp_train, yp_test = train_test_split(
    X, y_pass, test_size=0.2, random_state=42, stratify=y_pass
)
clf = Pipeline([("prep", preprocess), ("model", LogisticRegression(max_iter=1000))])
clf.fit(Xp_train, yp_train)
pred_pass = clf.predict(Xp_test)
joblib.dump(clf, OUT / "pass_fail_model.joblib")

summary = {
    "dataset_rows": int(len(df)),
    "marks_model_mae": round(float(mean_absolute_error(ym_test, pred_marks)), 3),
    "marks_model_r2": round(float(r2_score(ym_test, pred_marks)), 3),
    "pass_fail_accuracy": round(float(accuracy_score(yp_test, pred_pass)), 3),
    "pass_fail_report": classification_report(yp_test, pred_pass, output_dict=True, zero_division=0)
}
(OUT / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({k:v for k,v in summary.items() if k != "pass_fail_report"}, indent=2))

# KNN is included for similarity-based classification; K-Means provides exploratory groups.
knn = Pipeline([("prep", preprocess), ("model", KNeighborsClassifier(n_neighbors=5))])
knn.fit(Xp_train, yp_train)
joblib.dump(knn, OUT / "knn_pass_fail_model.joblib")
cluster_pipe = Pipeline([("prep", preprocess), ("model", KMeans(n_clusters=3, random_state=42, n_init=10))])
cluster_pipe.fit(X)
joblib.dump(cluster_pipe, OUT / "student_segments_model.joblib")
