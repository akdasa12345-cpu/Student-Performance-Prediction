# PBL 1: Student Performance Prediction and Academic Risk Analysis
**Section:** C | **Level:** MCA

## Overview
This project analyses student academic performance and predicts final examination marks, Pass/Fail status, and an academic risk category using attendance, study hours, previous marks, assignment marks, and internal assessment marks.

## Project structure
- `data/student_performance_demo.csv` — small illustrative dataset generated for demonstration.
- `src/train.py` — preprocessing, model training, evaluation, and model saving.
- `src/predict.py` — prediction for a new student record.
- `report/Project_Report.md` — project documentation.
- `requirements.txt` — Python dependencies.

## Dataset note
The included CSV is **synthetic demonstration data**, not real student records and not an official college dataset. Replace it with an approved, anonymized dataset before reporting real findings.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
python src/predict.py
```

The training script writes model files and evaluation output into `artifacts/`. Reported metrics are generated only when the script is run; no performance results have been invented in this project package.
