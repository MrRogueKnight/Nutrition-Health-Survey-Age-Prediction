# 🧠 Nutrition Health Survey – Age Prediction  
---
## 🎓 Summer Analytics 2025 – IIT Guwahati

*Organized by the Consulting and Analytics Club, IIT Guwahati*

📊 **Project:** Nutrition & Health Survey – Age Group Prediction

🔗 **Kaggle Notebook:** [View on Kaggle](https://www.kaggle.com/code/mrrogueknight/nutrition-health-survey-age-prediction)

---

You can copy-paste this into your README file or the Kaggle notebook description. Let me know if you'd like to add sections like:

* Overview / Problem Statement
* Approach / Models Used
* Results
* Future Improvements

I'm happy to help flesh it out!


---

## 📋 Overview

This challenge is based on a simplified subset of the **National Health and Nutrition Examination Survey (NHANES)** — a nationwide study conducted by the **CDC's National Center for Health Statistics**. NHANES uniquely combines **interviews, physical exams, and lab tests** to evaluate the health and nutrition of people in the U.S.

You're provided with a focused dataset of **selected health indicators** for over 2,300 individuals. Your task is to develop a **binary classification model** to predict whether a person is a **Senior (65+ years old)** or an **Adult (under 65 years)** based on these features.

---

## 🧠 Problem Statement

Build a binary classifier to predict the `age_group` of individuals based on their health profile.

- `age_group = 0` → Adult (under 65 years)
- `age_group = 1` → Senior (65 years and above)

Note: In the training data, this field is stored as text — `'Adult'` and `'Senior'`. You'll need to map them to integers before modeling:

```python
'Adult' → 0  
'Senior' → 1
```

---

## 📁 Files Provided

| File                    | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `Train_Data.csv`        | 2,016 rows with 7 features and the target column `age_group`                |
| `Test_Data.csv`         | 312 rows with 7 features but no target column                               |
| `Sample_Submission.csv` | Example format for your submission (`SEQN`, `age_group`)                    |

> 📝 Use the training data to build and validate your model. Then use the test data for predictions.

---

## 🔍 Feature Description

| Column     | Description                                                                 |
|------------|-----------------------------------------------------------------------------|
| `SEQN`     | Unique identifier for each respondent                                       |
| `RIAGENDR` | Gender (1 = Male, 2 = Female)                                                |
| `PAQ605`   | Physical activity (moderate/vigorous activity in a typical week)            |
| `BMXBMI`   | Body Mass Index                                                             |
| `LBXGLU`   | Glucose Level                                                                |
| `DIQ010`   | Diabetes questionnaire response                                              |
| `LBXGLT`   | Oral Glucose Tolerance                                                      |
| `LBXIN`    | Insulin Level                                                               |

> ⚠️ Missing values (`NaN`) may be present. Handle them with imputation or removal.

---

## 🎯 Target Variable – `age_group`

Your task is to predict this column for each entry in `Test_Data.csv`.

| Value | Description         |
|--------|----------------------|
| `0`    | Adult (under 65)     |
| `1`    | Senior (65 and above)|

✅ Your final predictions must contain **only** 0 or 1 values.

---

## 🧪 Evaluation Metric – F1 Score

Submissions are evaluated using the **F1 Score**, which balances precision and recall — ideal for imbalanced classes.

**Formula:**

> F1 = 2 × (Precision × Recall) / (Precision + Recall)

Where:
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)

📘 Learn more: [https://en.wikipedia.org/wiki/F1_score](https://en.wikipedia.org/wiki/F1_score)

---

## ⚙️ How the Challenge Works

1. **Train** your model on `Train_Data.csv`.
2. **Predict** `age_group` for `Test_Data.csv`.
3. **Submit** your results in the exact format below:

```csv
SEQN,age_group
12345,0
67890,1
...
```

4. **Leaderboard**:
   - **Public Leaderboard**: Based on ~50% of the test set.
   - **Private Leaderboard**: Based on the remaining 50%.

5. **Tie-Breaker**:  
   In case of a tie, the top-5 participants will be judged based on the **quality of their Feature Engineering and EDA**.

6. ✅ **Mark your best submission as FINAL** to be considered for private leaderboard rankings.

---

## ✅ Submission Checklist

- [x] Train your model using `Train_Data.csv`
- [x] Predict `age_group` (0 or 1) for all entries in `Test_Data.csv`
- [x] Submit a CSV with exactly two columns: `SEQN`, `age_group`
- [x] Ensure `age_group` predictions are integers (0 or 1 only)
- [x] Mark your best submission as **FINAL**

---

## 🧼 Data Notes & Modeling Tips

- Handle missing values properly (`mean`, `median`, or more advanced methods)
- Feature scaling may help, especially for glucose and insulin levels
- Try different models:
  - Logistic Regression
  - Random Forest
  - XGBoost (✅ officially allowed)
- Cross-validate your model (e.g., Stratified K-Fold)
- Explore class imbalance handling (`scale_pos_weight`, `class_weight`, SMOTE)

---

## 📝 Disclaimer

This dataset is derived from the NHANES data provided by the CDC. It has been preprocessed and simplified for educational use.  
We do **not** claim ownership of the original dataset.

---

## 👨‍💻 Maintainers

Hosted by:  
**Consulting and Analytics Club**  
**Indian Institute of Technology (IIT) Guwahati**

---

## 📚 References

- [NHANES Official Website](https://www.cdc.gov/nchs/nhanes)
- [F1 Score – Wikipedia](https://en.wikipedia.org/wiki/F1_score)

---

## 📌 License

This project is intended for **educational purposes only**. Please refer to NHANES’ official licensing terms for external use.
