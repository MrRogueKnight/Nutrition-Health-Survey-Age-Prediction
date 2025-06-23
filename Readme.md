# 🧠 Nutrition Health Survey - Age Prediction

### Summer Analytics 2025, IIT Guwahati

*Organized by the Consulting and Analytics Club, IIT Guwahati*

-----

## 📋 About This Project

The National Health and Nutrition Examination Survey (NHANES) is a nationally representative health study conducted by the CDC’s National Center for Health Statistics. It uniquely combines interviews, physical exams, and lab tests to assess the health and nutritional status of U.S. children and adults.

This dataset is a **subset, focused part of the NHANES study by CDC**, which looks at health and nutrition trends in the U.S. It has **6,287 entries and 7 key features**, covering things like body stats, lifestyle, and lab results. The task is simple: predict if a person is a senior (65+) or not. NHANES collected data through home visits, mobile clinics, and lab tests, giving a good mix of real and reported info. This trimmed version keeps only what’s needed, making it perfect for health-based age prediction.

This Hackathon is in collaboration with **Consulting and Analytics Club, IIT Guwahati** for **Summer Analytics 2025**.

-----

## 🧾 Dataset Description

The data contains two primary files: `train.csv` and `test.csv`.

  - **`train.csv`**: This file contains the training set observations, comprising **2,016 rows**. It includes all 7 features along with the target variable (`age_group`), which should be used for training your model.
  - **`test.csv`**: This file contains the testing set observations, with **312 rows**. It includes the same 7 features as `train.csv`, but the target column (`age_group`) is missing. You will use your trained model to predict these missing `age_group` values.

Additionally, a **`sample-submission.csv`** file is provided to illustrate the required format for your submission. Note that the header must include 'age\_group' as the column name, and the number of rows should match the number of rows in the test set.

### Target Variable: `age_group`

The task is to predict whether a person is a Senior (65+ years old) or an Adult (\<65 years old).
\*\*It is crucial to map 'Adult' to `0` and 'Senior`to`1`in your predictions.** Your submission file will only accept values`0`or`1\`.

| Label      | Meaning |
|-----------|---------|
| `0`       | Adult (\< 65 years) |
| `1`       | Senior (65+ years) |

### Features:

The dataset contains the following features:

| Column    | Description |
|------------|-------------|
| `SEQN`     | Sequence number (identifier) |
| `RIAGENDR` | Respondent's Gender (1 = Male, 2 = Female) |
| `PAQ605`   | Physical activity questionnaire response: If the respondent engages in moderate or vigorous-intensity sports, fitness, or recreational activities in the typical week |
| `BMXBMI`   | Body Mass Index |
| `LBXGLU`   | Glucose level |
| `DIQ010`   | Diabetes questionnaire response |
| `LBXGLT`   | Glucose tolerance (Oral) |
| `LBXIN`    | Insulin level |

> ⚠️ **Note on Missing Values:** The dataset may contain missing values (`NaN`) which should be handled appropriately during your data preprocessing.

-----

## 🧠 Objective

Build a **classification model** to predict whether a person is a **Senior (1)** or an **Adult (0)** based on health indicators. Use `train.csv` for training and make predictions on `test.csv`.

-----

## 📊 Evaluation Metric

Submissions are evaluated using the **F1 Score**.

> F1 Score balances **Precision** and **Recall**, making it ideal for imbalanced classes.

For clarity, here's how the F1 Score is calculated:

-----

## 🚀 Submission Guidelines

1.  Train your model using `train.csv`.
2.  Predict the `age_group` for each entry in `test.csv`.
3.  Ensure your output format matches `sample-submission.csv`. Your submission file should have two columns: `SEQN` and `age_group`.
4.  Only labels `0` or `1` are accepted in the `age_group` column.
5.  Submit your `.csv` file through the platform.

Here's an example of the expected submission format:

> ✅ **Make sure to mark your submission as FINAL** to appear on the **Private Leaderboard**.

-----

## 🧪 Evaluation Procedure

  - **Public Leaderboard**: Evaluated on \~50% of the test set.
  - **Private Leaderboard**: Final score calculated on the remaining 50%.
  - **In case of tied scores, the top-5 participants will be further evaluated and ranked based on their Feature Engineering and Exploratory Data Analysis (EDA) approach.**

-----

## 📝 Disclaimer

This dataset is derived and pre-processed from the original NHANES data by the CDC. It has been modified for educational purposes in this hackathon, focusing on classification, EDA, and feature engineering skills.

We do not claim ownership over the original dataset.

-----

## 👨‍💻 Maintainers & Credits

Organized by the **Consulting and Analytics Club**,  
**Indian Institute of Technology (IIT) Guwahati**

-----

## 📚 References

  - NHANES Data: [https://www.cdc.gov/nchs/nhanes](https://www.cdc.gov/nchs/nhanes)
  - F1 Score: [Wikipedia - F1 Score](https://en.wikipedia.org/wiki/F1_score)

-----

## 📌 License

This project is intended for **educational use only**. Please refer to the original NHANES dataset license for data usage terms.