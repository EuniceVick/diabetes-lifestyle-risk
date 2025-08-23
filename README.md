# Diabetes Risk Prediction from Lifestyle

This repository contains an **end‑to‑end ML project** for predicting diabetes risk from lifestyle and health factors.

## Contents
- `data/cleaned_diabetes_lifestyle.csv` — preprocessed dataset (target `Diabetic` as 0/1)
- `notebooks/diabetes_lifestyle_ml_explainer.ipynb` — complete workflow:
  - Data loading & Matplotlib‑only EDA
  - 5‑fold cross‑validated model comparison (LR, RF, GB, SVM)
  - Train best model, decision threshold tuning, ROC & PR curves
  - Probability calibration (Platt) + Brier score
  - SHAP global & local explanations
  - SHAP interaction values (incl. BMI × Activity)
- `models/diabetes_lifestyle_pipeline.pkl` — trained pipeline (preprocess + best model)
- `models/diabetes_lifestyle_pipeline_info.json` — metadata and thresholds
- `reports/shap_values_randomforest.csv` — SHAP values for test set
- `src/` — minimal scripts for inference and explanation

## Quickstart
```bash
pip install -r requirements.txt
jupyter notebook notebooks/diabetes_lifestyle_ml_explainer.ipynb
```

### Reuse the trained pipeline
```python
import joblib, pandas as pd
pipe = joblib.load('models/diabetes_lifestyle_pipeline.pkl')
df = pd.read_csv('data/cleaned_diabetes_lifestyle.csv')
X = df.drop(columns=['Diabetic'])
proba = pipe.predict_proba(X)[:, 1]  # risk score
pred = (proba >= 0.5).astype(int)    # default threshold
```

## Scripts
- `src/predict.py`: batch predict from CSV and save scores
- `src/explain.py`: save SHAP global bar plot and two local waterfalls
- `src/train.py`: re‑train pipeline and save artifacts

## License
MIT
