
# 🧠 Nutrition Health Survey – Age Prediction

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue.svg)](https://www.kaggle.com/code/mrrogueknight/nutrition-health-survey-age-prediction)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

Predict age group (Adult vs Senior) using health indicators from the **National Health and Nutrition Examination Survey (NHANES)** dataset.

---

## 📊 Leaderboard Rankings

| Leaderboard | Rank | Out of |
|-------------|------|--------|
| 🔒 Private | **32** | 6,900+ |
| 🧪 Practice | **30** | 6,900+ |
| 🌐 Public | **55** | 6,900+ |

---

## 🎯 Problem Statement

Build a **binary classification model** to predict whether an individual is a **Senior (65+ years)** or an **Adult (under 65)** based on health indicators.

- `age_group = 0` → Adult (under 65)
- `age_group = 1` → Senior (65+)

---

## 📁 Dataset

### Features

| Column | Description |
|--------|-------------|
| `SEQN` | Unique respondent ID |
| `RIAGENDR` | Gender (1 = Male, 2 = Female) |
| `PAQ605` | Physical activity level |
| `BMXBMI` | Body Mass Index |
| `LBXGLU` | Glucose Level |
| `DIQ010` | Diabetes questionnaire response |
| `LBXGLT` | Oral Glucose Tolerance |
| `LBXIN` | Insulin Level |

### Files
- `Train_Data.csv` - 2,016 rows with labels
- `Test_Data.csv` - 312 rows (no labels)

---

## 🛠️ Approach

### Feature Engineering
- **BMI Categories** - Binned BMI into groups (Underweight, Normal, Overweight, Obese)
- **High Glucose Flag** - Binary indicator for glucose > 125 mg/dL
- **Glucose-Insulin Interaction** - Multiplication of glucose and insulin levels

### Models Used
- **XGBoost** with `scale_pos_weight` for class imbalance
- **LightGBM** with `class_weight='balanced'`
- **Ensemble** - Averaged probabilities from both models

### Pipeline
```
Raw Data → Feature Engineering → Imputation → Scaling → Model Training → Prediction
```

### Cross-Validation
- **Stratified K-Fold** (5 folds)
- **Threshold Tuning** using Precision-Recall curve for optimal F1 score

---

## 📈 Results

**Evaluation Metric:** F1 Score

### Feature Importance
Both models identified key predictors:
- `LBXGLU` (Glucose Level)
- `LBXIN` (Insulin Level)
- `BMXBMI` (BMI)
- `DIQ010` (Diabetes Status)

---

## 🚀 Quick Start

### Clone the Repository
```bash
git clone https://github.com/MrRogueKnight/Nutrition-Health-Survey-Age-Prediction.git
cd Nutrition-Health-Survey-Age-Prediction
```

### Install Dependencies
```bash
pip install numpy pandas scikit-learn xgboost lightgbm matplotlib seaborn
```

### Run the Pipeline
```python
python train.py
```

Or use the [Kaggle Notebook](https://www.kaggle.com/code/mrrogueknight/nutrition-health-survey-age-prediction) directly.

---

## 📂 Repository Structure

```
Nutrition-Health-Survey-Age-Prediction/
│
├── train.py                    # Main training script
├── final_submission.csv        # Final predictions
├── xgb_model.joblib            # Trained XGBoost model
├── lgb_model.joblib            # Trained LightGBM model
├── README.md                   # This file
└── LICENSE                     # MIT License
```

---

## 👥 Contributors

| Prashant Ranjan | Uday Tripathi |
|-----------------|---------------|
| Project Lead | Co-Developer |
| Mathematics & Computing at RGIPT | VNR VJIET, Hyderabad · Minor in ML (IIIT-H) |
| [GitHub](https://github.com/MrRogueKnight) · [LinkedIn](https://www.linkedin.com/in/mrrogueknight/) | [GitHub](https://github.com/udaytripathi51) · [LinkedIn](https://www.linkedin.com/in/uday-tripathi51/) |
| ⚡ 50% Contribution | ⚡ 50% Contribution |

**🤝 Equal Collaboration:** Both contributors worked equally on feature engineering, model development, hyperparameter tuning, and documentation.

---

## 📚 References

- [NHANES Official Website](https://www.cdc.gov/nchs/nhanes)
- [F1 Score – Wikipedia](https://en.wikipedia.org/wiki/F1_score)

---

## 📄 License

This project is for **educational purposes** and is licensed under the [MIT License](./LICENSE).

---

> Built with ❤️ during Summer Analytics 2025 at IIT Guwahati
