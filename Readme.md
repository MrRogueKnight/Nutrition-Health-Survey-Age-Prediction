
---

# 🧠 Nutrition Health Survey – Age Prediction  
**Summer Analytics 2025 | Hosted by Consulting and Analytics Club, IIT Guwahati**  

---

## 📋 Overview  
This competition uses a curated subset of the **National Health and Nutrition Examination Survey (NHANES)**—a CDC-led study combining interviews, physical exams, and lab tests to assess U.S. population health.  

**Your Task:**  
Build a binary classification model to predict whether a person is a **Senior (65+ years)** or an **Adult (<65 years)** using **7 key health features** from 6,287 entries.  

---

## 🎯 Problem Statement  
**Objective:** Predict `age_group` (binary classification):  
- **`0`** → Adult (<65 years)  
- **`1`** → Senior (≥65 years)  

**Evaluation Metric:** **F1 Score** (balances precision and recall).  

---

## 📂 Dataset Files  
| File                   | Description                                  | Rows  | Columns |
|------------------------|----------------------------------------------|-------|---------|
| `Train_Data.csv`       | Labeled training data (features + `age_group`) | 2,016 | 8       |
| `Test_Data.csv`        | Unlabeled test data (features only)          | 312   | 7       |
| `Sample_Submission.csv`| Submission format (`SEQN`, `age_group`)      | -     | 2       |

---

## 🔍 Feature Descriptions  
| Feature    | Description                                  | Notes               |
|------------|----------------------------------------------|---------------------|
| `SEQN`     | Unique respondent ID                         | Identifier          |
| `RIAGENDR` | Gender (`1`: Male, `2`: Female)              | Categorical         |
| `PAQ605`   | Moderate/vigorous physical activity (weekly) | Likert-scale?       |
| `BMXBMI`   | Body Mass Index (BMI)                        | Continuous          |
| `LBXGLU`   | Glucose level (mg/dL)                        | May have `NaN`      |
| `DIQ010`   | Diabetes diagnosis (self-reported)           | Binary/Categorical? |
| `LBXGLT`   | Oral Glucose Tolerance Test result           | Continuous          |
| `LBXIN`    | Insulin level (µIU/mL)                       | Continuous          |

⚠️ **Note:** Handle missing values (`NaN`) during preprocessing.  

---

## 📊 Evaluation  
**Metric:** **F1 Score** (Harmonic mean of precision and recall):  
\[
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]  
**Where:**  
- **Precision** = `TP / (TP + FP)`  
- **Recall** = `TP / (TP + FN)`  

**Leaderboard:**  
- **Public (50% test data)** → Preliminary rankings.  
- **Private (50% test data)** → Final rankings (*must mark submission as FINAL*).  

**Tiebreaker:** Quality of EDA/feature engineering.  

---

## ⚙️ Workflow  
1. **Train**  
   - Use `Train_Data.csv` to build/validate models.  
2. **Predict**  
   - Generate `age_group` predictions for `Test_Data.csv`.  
3. **Submit**  
   - Follow `Sample_Submission.csv` format:  
     ```csv
     SEQN, age_group
     12345, 0
     67890, 1
     ```  

---

## 🛠 Best Practices  
- **Data Cleaning:** Impute/remove `NaN` values.  
- **Feature Scaling:** Normalize continuous variables (e.g., glucose, insulin).  
- **EDA:** Visualize distributions/correlations (e.g., boxplots for BMI vs. age).  
- **Models to Try:**  
  - Logistic Regression  
  - Random Forest/XGBoost  
  - SVM (with kernel tuning)  
- **Validation:** Use k-fold cross-validation.  

---

## 📌 Rules & Notes  
- **Submission Limits:** 5 entries/day.  
- **Final Submission:** Must be marked **FINAL** for private LB eligibility.  
- **Team Size:** Max 3 members.  
- **License:** NHANES data is CDC-owned; this challenge is for educational use.  

---

## 👥 Organizers  
**Consulting and Analytics Club**  
Indian Institute of Technology (IIT) Guwahati  

📚 **References:**  
- [NHANES Official Website](https://www.cdc.gov/nchs/nhanes)  
- [F1 Score Explained](https://en.wikipedia.org/wiki/F1_score)  

---

### 🚀 Ready to Compete?  
1. Download the dataset.  
2. Train your model.  
3. Submit predictions!  

**Good luck!**  

--- 
