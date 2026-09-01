# Mermaid Diagrams for Nutrition Health Survey Project

---

## 1. Complete Project Pipeline

```mermaid
flowchart TD
    A["Start"] --> B["Load Data"]
    B --> C["Train_Data.csv 1,966 rows"]
    B --> D["Test_Data.csv 312 rows"]
    
    C --> E["Data Cleaning"]
    D --> E
    
    E --> F["Remove rows with missing SEQN -12 rows"]
    F --> G["Remove rows with missing age_group -14 rows"]
    G --> H["Final Training Data 1,940 rows"]
    
    H --> I["Handle Missing Values"]
    I --> J["Fill with Medians: RIAGENDR to 2.0, PAQ605 to 2.0, BMXBMI to 26.8, LBXGLU to 97.0, DIQ010 to 2.0, LBXGLT to 105.0, LBXIN to 9.01"]
    
    J --> K["Feature Engineering 7 to 48 Features"]
    
    K --> L["Polynomial Features BMI squared, Glucose squared, Insulin squared"]
    K --> M["Log and Sqrt Transforms log(BMI), sqrt(Glucose)"]
    K --> N["Interaction Features Glu by Insulin, BMI by Glucose"]
    K --> O["Categorical Features BMI Categories, Glucose Categories"]
    K --> P["Clinical Scores Diabetes Risk, HOMA-IR"]
    K --> Q["Outlier Indicators Extreme Value Flags"]
    
    L --> R["Preprocessing"]
    M --> R
    N --> R
    O --> R
    P --> R
    Q --> R
    
    R --> S["Imputation Fill remaining NaNs"]
    S --> T["Scaling StandardScaler"]
    T --> U["Feature Selection Select Top 20 of 48"]
    
    U --> V["Split Data Training 90 percent Validation 10 percent"]
    
    V --> W["Train XGBoost n_estimators 400 learning_rate 0.07 max_depth 4"]
    V --> X["Train LightGBM n_estimators 400 learning_rate 0.07 class_weight balanced"]
    V --> Y["Train CatBoost iterations 500 depth 5 learning_rate 0.05"]
    
    W --> Z["Cross-Validation 5-Fold CV"]
    X --> Z
    Y --> Z
    
    Z --> AA["Threshold Tuning Optimize F1 Score"]
    AA --> AB["CV Results: XGBoost 0.4196, LightGBM 0.4224, CatBoost 0.4184"]
    
    AB --> AC["Build Ensemble Stacking Classifier"]
    AC --> AD["Meta Learner Logistic Regression"]
    AD --> AE["Stacking CV F1 0.3972"]
    
    AE --> AF["Train on Full Dataset"]
    AF --> AG["Make Predictions 310 Test Samples"]
    
    AG --> AH["Apply Threshold Ensemble Threshold 0.31"]
    AH --> AI["Final Results 229 Adults, 81 Seniors"]
    AI --> AJ["Generate Submission"]
    AJ --> AK["End"]
    
    style A fill:#2ecc71,color:#fff
    style AK fill:#e74c3c,color:#fff
    style W fill:#3498db,color:#fff
    style X fill:#3498db,color:#fff
    style Y fill:#3498db,color:#fff
    style AI fill:#f39c12,color:#fff
```

---

## 2. Feature Engineering Details

```mermaid
flowchart LR
    subgraph RAW["Raw Features 7"]
        R1["RIAGENDR Gender"]
        R2["PAQ605 Activity"]
        R3["BMXBMI BMI"]
        R4["LBXGLU Glucose"]
        R5["DIQ010 Diabetes"]
        R6["LBXGLT Tolerance"]
        R7["LBXIN Insulin"]
    end
    
    subgraph POLY["Polynomial Features"]
        P1["BMI squared, BMI cubed"]
        P2["Glucose squared, Glucose cubed"]
        P3["Insulin squared, Insulin cubed"]
        P4["Tolerance squared, Tolerance cubed"]
    end
    
    subgraph TRANSFORM["Transform Features"]
        T1["log(BMI), sqrt(BMI)"]
        T2["log(Glucose), sqrt(Glucose)"]
        T3["log(Insulin), sqrt(Insulin)"]
        T4["log(Tolerance), sqrt(Tolerance)"]
    end
    
    subgraph INTERACTION["Interaction Features"]
        I1["Glucose times Insulin"]
        I2["BMI times Glucose"]
        I3["BMI times Insulin"]
        I4["Gender times BMI"]
        I5["Glucose divided by Insulin"]
    end
    
    subgraph CATEGORICAL["Categorical Features"]
        C1["BMI Category 0-3"]
        C2["Glucose Category 0-2"]
        C3["Insulin High 0-1"]
        C4["Obesity Indicator 0-1"]
    end
    
    subgraph CLINICAL["Clinical Scores"]
        CL1["Diabetes Risk"]
        CL2["HOMA-IR Index"]
        CL3["Composite Risk Score"]
    end
    
    subgraph OUTLIER["Outlier Indicators"]
        O1["Glucose Outlier"]
        O2["Insulin Outlier"]
        O3["BMI Outlier"]
        O4["Tolerance Outlier"]
    end
    
    subgraph MISSING["Missing Indicators"]
        M1["RIAGENDR Missing"]
        M2["BMXBMI Missing"]
        M3["LBXGLU Missing"]
        M4["LBXIN Missing"]
    end
    
    RAW --> POLY
    RAW --> TRANSFORM
    RAW --> INTERACTION
    RAW --> CATEGORICAL
    RAW --> CLINICAL
    RAW --> OUTLIER
    RAW --> MISSING
    
    POLY --> SELECTED["Selected Features Top 20 of 48"]
    TRANSFORM --> SELECTED
    INTERACTION --> SELECTED
    CATEGORICAL --> SELECTED
    CLINICAL --> SELECTED
    OUTLIER --> SELECTED
    MISSING --> SELECTED
    
    style RAW fill:#3498db,color:#fff
    style SELECTED fill:#2ecc71,color:#fff
```

---

## 3. Model Training and Cross-Validation

```mermaid
flowchart TD
    A["Data Preprocessed 20 Features Selected"] --> B["Split Data Stratified 5-Fold CV"]
    
    B --> C["Fold 1 Train 80 percent Val 20 percent"]
    B --> D["Fold 2 Train 80 percent Val 20 percent"]
    B --> E["Fold 3 Train 80 percent Val 20 percent"]
    B --> F["Fold 4 Train 80 percent Val 20 percent"]
    B --> G["Fold 5 Train 80 percent Val 20 percent"]
    
    C --> H["Train XGBoost"]
    C --> I["Train LightGBM"]
    C --> J["Train CatBoost"]
    
    H --> K["Predict on Validation"]
    I --> L["Predict on Validation"]
    J --> M["Predict on Validation"]
    
    K --> N["Calculate F1 Score"]
    L --> O["Calculate F1 Score"]
    M --> P["Calculate F1 Score"]
    
    N --> Q["Tune Threshold Optimize F1"]
    O --> R["Tune Threshold Optimize F1"]
    P --> S["Tune Threshold Optimize F1"]
    
    Q --> T["XGBoost Results F1 0.4196 Threshold 0.252"]
    R --> U["LightGBM Results F1 0.4224 Threshold 0.314"]
    S --> V["CatBoost Results F1 0.4184 Threshold 0.183"]
    
    T --> W["Build Ensemble Stacking Classifier"]
    U --> W
    V --> W
    
    W --> X["Train Meta Model Logistic Regression"]
    X --> Y["Ensemble Results F1 0.3972"]
    
    Y --> Z["Validate on Holdout Set 10 percent of Data"]
    Z --> AA["Final Performance F1 0.3846"]
    
    style T fill:#3498db,color:#fff
    style U fill:#2ecc71,color:#fff
    style V fill:#e67e22,color:#fff
    style Y fill:#9b59b6,color:#fff
    style AA fill:#f1c40f,color:#000
```

---

## 4. Stacking Ensemble Architecture

```mermaid
flowchart TD
    subgraph INPUT["Input Data"]
        I["Test Sample 310 rows"]
    end
    
    subgraph LEVEL1["Level 1 Base Models"]
        X["XGBoost n_estimators 400 max_depth 4"]
        L["LightGBM n_estimators 400 class_weight balanced"]
        C["CatBoost iterations 500 depth 5"]
    end
    
    subgraph PREDICTIONS["Base Model Predictions"]
        XP["XGBoost Probability 0.65"]
        LP["LightGBM Probability 0.60"]
        CP["CatBoost Probability 0.55"]
    end
    
    subgraph LEVEL2["Level 2 Meta Model"]
        M["Meta Learner Logistic Regression C 1.0 class_weight balanced"]
    end
    
    subgraph OUTPUT["Final Output"]
        O["Final Prediction Probability 0.62 Threshold 0.31 to SENIOR"]
    end
    
    I --> X
    I --> L
    I --> C
    
    X --> XP
    L --> LP
    C --> CP
    
    XP --> M
    LP --> M
    CP --> M
    
    M --> O
    
    style X fill:#3498db,color:#fff
    style L fill:#2ecc71,color:#fff
    style C fill:#e67e22,color:#fff
    style M fill:#9b59b6,color:#fff
    style O fill:#e74c3c,color:#fff
```

---

## 5. Data Flow Diagram

```mermaid
flowchart LR
    subgraph SOURCES["Data Sources"]
        S1["Train_Data.csv 1,966 rows"]
        S2["Test_Data.csv 312 rows"]
    end
    
    subgraph PROCESS1["Process 1 Data Cleaning"]
        P1["Remove Invalid Rows Drop SEQN NaN Drop age_group NaN"]
    end
    
    subgraph PROCESS2["Process 2 Imputation"]
        P2["Fill Missing Values with Medians"]
    end
    
    subgraph PROCESS3["Process 3 Feature Engineering"]
        P3["7 to 48 Features Polynomial, Log, Interaction, Categorical, Clinical Scores"]
    end
    
    subgraph PROCESS4["Process 4 Preprocessing"]
        P4["Imputation Scaling Feature Selection"]
    end
    
    subgraph PROCESS5["Process 5 Model Training"]
        P5["XGBoost LightGBM CatBoost Stacking Ensemble"]
    end
    
    subgraph OUTPUTS["Outputs"]
        O1["Model Files joblib pkl"]
        O2["Submission.csv 310 predictions"]
        O3["Training Logs"]
        O4["Config.json"]
    end
    
    S1 --> P1
    S2 --> P4
    
    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> P5
    
    P5 --> O1
    P5 --> O2
    P5 --> O3
    P5 --> O4
    
    style S1 fill:#3498db,color:#fff
    style S2 fill:#3498db,color:#fff
    style P5 fill:#2ecc71,color:#fff
    style O2 fill:#f39c12,color:#fff
```

---

## 6. Class Imbalance and Handling Strategy

```mermaid
flowchart TD
    subgraph PROBLEM["The Problem Class Imbalance"]
        A["Training Data 1,940 samples"]
        A --> B["Adult 84 percent 1,627 samples"]
        A --> C["Senior 16 percent 313 samples"]
    end
    
    subgraph ISSUES["Why This is an Issue"]
        D["Naive Model Always Predict Adult"]
        D --> E["Accuracy 84 percent Looks Good"]
        D --> F["But Finds 0 Seniors F1 Score 0.00"]
    end
    
    subgraph STRATEGIES["Handling Strategies"]
        G["Strategy 1 Class Weights"]
        H["Strategy 2 Threshold Tuning"]
        I["Strategy 3 F1 Score Optimization"]
    end
    
    subgraph RESULT["The Result"]
        J["Balanced Model"]
        J --> K["Accuracy 72 percent Lower Overall"]
        J --> L["Finds 40 percent of Seniors F1 Score 0.38"]
        J --> M["138 percent Improvement Over Random"]
    end
    
    PROBLEM --> ISSUES
    ISSUES --> STRATEGIES
    STRATEGIES --> RESULT
    
    style B fill:#2ecc71,color:#fff
    style C fill:#e74c3c,color:#fff
    style E fill:#f39c12,color:#fff
    style F fill:#e74c3c,color:#fff
    style L fill:#2ecc71,color:#fff
    style M fill:#2ecc71,color:#fff
```

---

## 7. Decision Tree Visualization

```mermaid
flowchart TD
    A["Start New Patient Data"] --> B["Glucose greater than 110"]
    
    B -->|Yes| C["Insulin greater than 20"]
    B -->|No| D["BMI greater than 32"]
    
    C -->|Yes| E["BMI greater than 28"]
    C -->|No| F["Predict Adult Confidence 70 percent"]
    
    E -->|Yes| G["Predict Senior Confidence 85 percent"]
    E -->|No| H["Predict Senior Confidence 65 percent"]
    
    D -->|Yes| I["Diabetes equals Yes"]
    D -->|No| J["Predict Adult Confidence 90 percent"]
    
    I -->|Yes| K["Predict Senior Confidence 60 percent"]
    I -->|No| L["Predict Adult Confidence 75 percent"]
    
    style G fill:#e74c3c,color:#fff
    style H fill:#e74c3c,color:#fff
    style K fill:#e74c3c,color:#fff
    style F fill:#2ecc71,color:#fff
    style J fill:#2ecc71,color:#fff
    style L fill:#2ecc71,color:#fff
```

---

## 8. Performance Comparison

```mermaid
flowchart LR
    subgraph BASELINE["Baseline Models"]
        B1["Random Guessing F1 0.16"]
        B2["Always Adult F1 0.00"]
        B3["Decision Tree F1 0.20"]
        B4["Logistic Regression F1 0.25"]
    end
    
    subgraph OURS["Our Models"]
        O1["XGBoost F1 0.4196"]
        O2["LightGBM F1 0.4224"]
        O3["CatBoost F1 0.4184"]
        O4["Stacking Ensemble F1 0.3972"]
    end
    
    subgraph BEST["Best Performance"]
        BE["LightGBM F1 0.4224"]
        BE2["Validation F1 0.3846"]
        BE3["Rank 32 out of 6900+ Top 0.5 percent"]
    end
    
    B1 --> O1
    B2 --> O2
    B3 --> O3
    B4 --> O4
    
    O1 --> BE
    O2 --> BE
    O3 --> BE
    O4 --> BE
    
    BE --> BE2
    BE2 --> BE3
    
    style B1 fill:#95a5a6,color:#fff
    style B2 fill:#95a5a6,color:#fff
    style B3 fill:#95a5a6,color:#fff
    style B4 fill:#95a5a6,color:#fff
    style O2 fill:#2ecc71,color:#fff
    style BE fill:#2ecc71,color:#fff
    style BE3 fill:#f1c40f,color:#000
```

---

## 9. Complete System Architecture

```mermaid
flowchart TD
    subgraph DATA["Data Layer"]
        D1["Training Data NHANES Survey"]
        D2["Test Data Unlabeled"]
    end
    
    subgraph PROCESSING["Processing Layer"]
        P1["Data Cleaning and Validation"]
        P2["Feature Engineering 7 to 48 Features"]
        P3["Preprocessing Impute, Scale, Select"]
        P4["Model Training XGBoost, LightGBM, CatBoost"]
        P5["Ensemble Stacking Classifier"]
    end
    
    subgraph STORAGE["Storage Layer"]
        S1["Model Artifacts joblib files"]
        S2["Config and Metadata config.json"]
        S3["Submission Files csv"]
        S4["Logs log"]
    end
    
    subgraph OUTPUT["Output Layer"]
        O1["Predictions 229 Adult, 81 Senior"]
        O2["Performance Report F1 0.3846"]
        O3["Visualizations 10 Charts"]
    end
    
    D1 --> P1
    D2 --> P1
    
    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> P5
    
    P4 --> S1
    P5 --> S1
    
    P1 --> S4
    P2 --> S4
    P3 --> S4
    P4 --> S4
    P5 --> S4
    
    P5 --> S2
    P5 --> S3
    
    S3 --> O1
    S2 --> O2
    S4 --> O3
    
    style D1 fill:#3498db,color:#fff
    style D2 fill:#3498db,color:#fff
    style P5 fill:#2ecc71,color:#fff
    style O1 fill:#f39c12,color:#fff
```

---

## 10. Timeline and Milestones

```mermaid
flowchart LR
    subgraph PHASE1["Phase 1 Data Understanding Day 1-2"]
        A1["Explore Data Understand Features"]
        A2["Analyze Imbalance 84 percent / 16 percent"]
        A3["Identify Missing Values 7 columns with NaNs"]
    end
    
    subgraph PHASE2["Phase 2 Data Preparation Day 3-4"]
        B1["Clean Data Remove Invalid Rows"]
        B2["Impute Missing Median Strategy"]
        B3["Feature Engineering 7 to 48 Features"]
    end
    
    subgraph PHASE3["Phase 3 Model Development Day 5-7"]
        C1["Preprocessing Pipeline Scale, Select"]
        C2["Train XGBoost CV 0.4196"]
        C3["Train LightGBM CV 0.4224"]
        C4["Train CatBoost CV 0.4184"]
    end
    
    subgraph PHASE4["Phase 4 Ensemble and Tuning Day 8-9"]
        D1["Build Stacking Ensemble"]
        D2["Threshold Tuning Optimize F1"]
        D3["Final Validation F1 0.3846"]
    end
    
    subgraph PHASE5["Phase 5 Production Day 10"]
        E1["Generate Predictions 310 samples"]
        E2["Create Submission CSV"]
        E3["Documentation Complete Guide"]
    end
    
    PHASE1 --> PHASE2
    PHASE2 --> PHASE3
    PHASE3 --> PHASE4
    PHASE4 --> PHASE5
    
    style C3 fill:#2ecc71,color:#fff
    style D3 fill:#2ecc71,color:#fff
    style E2 fill:#f39c12,color:#fff
```

---

## Summary of Diagrams

| # | Diagram Name | Purpose |
|---|--------------|---------|
| 1 | Complete Pipeline | Overview of entire project workflow |
| 2 | Feature Engineering | Understanding feature creation process |
| 3 | Model Training | CV and threshold tuning process |
| 4 | Stacking Ensemble | Ensemble architecture explanation |
| 5 | Data Flow Diagram | System architecture overview |
| 6 | Class Imbalance | Problem identification and solution |
| 7 | Decision Tree | Model logic explanation |
| 8 | Performance Comparison | Results visualization |
| 9 | System Architecture | High-level architecture view |
| 10 | Timeline | Project milestones and phases |
