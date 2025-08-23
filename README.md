# Diabetes Risk Prediction from Lifestyle

End‑to‑end ML project for predicting diabetes risk from lifestyle and health features.  
Includes **EDA, cross‑validated modeling, probability calibration, and SHAP explainability (global, local, interactions)**.

---

## 🚀 Relevance
Diabetes is one of the fastest-growing global health challenges, with lifestyle playing a major role in both onset and management.  
Traditional approaches rely heavily on clinical biomarkers and invasive tests, leaving gaps for **early, accessible, and low-cost risk assessment**.  
This project bridges that gap by leveraging **everyday lifestyle indicators** — such as sleep, BMI, stress, and physical activity — to predict diabetes risk.  
By focusing on factors within personal control, the work aligns with **preventive healthcare** and empowers individuals to take proactive steps.

## 🌍 Impact
- **Healthcare Support**: Provides clinicians and health programs with an inexpensive tool for population-level screening.  
- **Public Health Planning**: Can inform policy makers about lifestyle trends linked to diabetes, supporting data-driven interventions.  
- **Education**: Encourages individuals to understand the role of lifestyle in chronic disease risk.  
- **Scalability**: The pipeline is reproducible, interpretable, and can be adapted to diverse populations or extended to other diseases.

## ✨ Novelty
- **Lifestyle-Centered Modeling**: Unlike conventional diabetes prediction that relies mostly on blood tests and clinical features, this project models **lifestyle behaviors directly**.  
- **Explainability Built-In**: Using **SHAP explainability**, we not only predict outcomes but also show **why** predictions are made, ensuring trust and transparency.  
- **Probability Calibration**: Beyond raw predictions, probabilities are **calibrated** to improve clinical usability and decision-making.  
- **Interaction Insights**: SHAP interaction analysis uncovers **synergistic effects** (e.g., BMI × Physical Activity), offering richer understanding than single-variable effects.  
- **Portfolio-Ready & Reproducible**: The entire workflow is packaged with notebook, scripts, and artifacts — allowing seamless reuse, demonstration, and extension.

## 🎯 Contribution
This project demonstrates how **machine learning can extend beyond lab data** into the real-world domain of daily habits, addressing the United Nations **SDG 3 (Good Health and Well-being)**.  
It’s not just a technical exercise — it’s a step towards **actionable AI in healthcare**, showing how predictive analytics can support preventive medicine and empower healthier lifestyles.

---

## 📂 Repository Structure
- `data/cleaned_diabetes_lifestyle.csv` — preprocessed dataset  
- `notebooks/diabetes_lifestyle_ml_explainer.ipynb` — complete workflow (EDA → CV → calibration → SHAP)  
- `models/` — saved pipeline & metadata after running the notebook  
- `reports/` — SHAP exports  
- `src/` — scripts (`train.py`, `predict.py`, `explain.py`)  
- `requirements.txt`, `.gitignore`, `LICENSE`  

---

## ⚡ Quickstart

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run notebook
```bash
jupyter notebook notebooks/diabetes_lifestyle_ml_explainer.ipynb
```

### Use trained pipeline
```python
import joblib, pandas as pd
pipe = joblib.load('models/diabetes_lifestyle_pipeline.pkl')
df = pd.read_csv('data/cleaned_diabetes_lifestyle.csv')
X = df.drop(columns=['Diabetic'])
proba = pipe.predict_proba(X)[:, 1]  # risk score
pred = (proba >= 0.5).astype(int)    # classification
```

---

## 📜 License
MIT License — free to use, modify, and share with attribution.

