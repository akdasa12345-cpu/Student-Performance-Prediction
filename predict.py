from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts" / "final_marks_model.joblib"

# Example student record. Replace values with a real, consented/anonymized record.
student = pd.DataFrame([{
    "attendance_pct": 82,
    "study_hours_per_week": 12,
    "previous_marks": 68,
    "assignment_marks": 75,
    "internal_marks": 70,
    "internet_access": "Yes"
}])

if not MODEL.exists():
    raise SystemExit("Model not found. Run: python src/train.py")
model = joblib.load(MODEL)
predicted = float(model.predict(student)[0])
risk = "High" if predicted < 40 else ("Medium" if predicted < 60 else "Low")
print(f"Predicted final examination marks: {predicted:.1f}")
print(f"Academic risk level: {risk}")
