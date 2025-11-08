# Project 2: Discovering Fraud Patterns with the KDD Process

This project uses the **Knowledge Discovery in Databases (KDD)** methodology to analyze the Credit Card Fraud Detection dataset. The goal is not just to build a model, but to "discover" and "interpret" the key patterns and factors that distinguish fraudulent transactions from legitimate ones.

**Dataset Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Output Directory:** `kdd_fraud_detection_outputs/`
- Text Output: `kdd_fraud_detection_output.txt`
- Visualizations: 8 high-quality PNG images (300 DPI)

---

## 1. Selection

The first step is to **select** the target data and variables for our knowledge discovery task.

* **Data Source:** We are using the "Credit Card Fraud Detection Dataset" from Kaggle, which contains transactions made by European cardholders in September 2013. We work with 284,807 transactions.
* **Goal:** To discover knowledge about what features and patterns most significantly distinguish fraudulent transactions from legitimate ones.
* **Target Variable:** Our target for discovery is the `Class` column (0 = Normal, 1 = Fraud).
* **Initial Feature Set:** The dataset contains 30 features:
    * **Time:** Seconds elapsed between this transaction and the first transaction
    * **V1-V28:** Principal Component Analysis (PCA) transformed features (anonymized for privacy)
    * **Amount:** Transaction amount
* **Data Characteristics:** The dataset is highly **imbalanced** with only **0.1727% fraud** cases (492 frauds out of 284,807 transactions).

---

## 2. Pre-processing

This phase involves "cleaning" the raw selected data to handle imperfections. The fraud detection dataset requires careful pre-processing to handle data quality issues.

* **Handling Missing Data:** After thorough analysis, we found **no missing values** in the dataset - all 284,807 records are complete.
* **Handling Duplicates:** We identified and removed **1,081 duplicate transactions**, reducing the dataset to 283,726 unique transactions.
* **Handling Outliers:** We detected **31,685 outlier transactions** (11.17% of data) with amounts exceeding $185.38. However, we **kept these outliers** as they may represent legitimate high-value transactions or actual fraud cases that are critical for our analysis.
* **Data Type Verification:** Confirmed all features are numerical (30 float64, 1 int64), which is appropriate for our analysis.

---

## 3. Transformation

With a clean dataset, we now **transform** it into a format suitable for the data mining algorithms.

* **Feature Engineering:** We created new, more powerful features from existing ones:
    * **Time-based features:**
        * `Hour` = (Time / 3600) % 24
        * `Day` = (Time / 86400)
        * `TimePeriod` = Categorical (Morning/Afternoon/Evening/Night)
    * **Amount-based features:**
        * `Amount_log` = log(Amount + 1)
        * `Amount_sqrt` = sqrt(Amount)
        * `Amount_Category` = Categorical (Zero/Small/Medium/Large/Very_Large)
    * **PCA interaction features:**
        * `V1_V2_interaction` = V1 × V2
        * `V1_V3_interaction` = V1 × V3
        * `PCA_sum` = Sum of all V1-V28 features
        * `PCA_mean` = Mean of all V1-V28 features
        * `PCA_std` = Standard deviation of all V1-V28 features

* **Encoding Categorical Data:** We used **One-Hot Encoding** to convert categorical features (`TimePeriod` and `Amount_Category`) into binary (0/1) format that our algorithms can process.

* **Normalizing Numerical Data:** We applied **RobustScaler** to all numerical features. This scaler is particularly effective for datasets with outliers, as it uses the median and interquartile range instead of mean and standard deviation.

* **Handling Class Imbalance:** We applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the training data:
    * Original imbalance ratio: **599:1** (Normal:Fraud)
    * After SMOTE: **2:1** (Normal:Fraud)
    * This increased fraud samples from 331 to 99,138 in the training set

* **Final Dataset Shape:** After all transformations, we have **45 features** (from original 30) with 198,608 training samples and 85,118 test samples (70-30 split).

---

## 4. Data Mining

This is the core "discovery" step where we apply algorithms to our transformed data to find patterns. Our task is **binary classification** (detecting fraud vs. normal transactions).

* **Algorithm 1 (Baseline):** We used **Logistic Regression** with balanced class weights. This is a fast, linear model that provides interpretable results and serves as our baseline.

* **Algorithm 2 (Tree-based):** We used a **Random Forest Classifier** with 100 trees and balanced class weights. This powerful ensemble model excels at finding complex, non-linear patterns and can **rank features by importance**, which is central to our knowledge discovery goal.

* **Algorithm 3 (Boosting):** We used a **Gradient Boosting Classifier** with 100 estimators. This advanced ensemble method builds trees sequentially, each correcting errors from the previous one, often achieving state-of-the-art performance.

* **Training/Test Split:** We split our data (70% training, 30% testing) to ensure we evaluate patterns on unseen data. The split was **stratified** to maintain the fraud ratio in both sets.

---

## 5. Interpretation & Evaluation

The final step is to **evaluate** the patterns we found and **interpret** them into human-readable "knowledge."

### Evaluation Metrics

We evaluated our models using multiple metrics appropriate for imbalanced classification:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 0.9684 | 0.0454 | 0.8944 | 0.0864 | 0.9544 |
| **Random Forest** | **0.9995** | **0.9310** | 0.7606 | **0.8372** | 0.9720 |
| **Gradient Boosting** | 0.9982 | 0.4797 | **0.8310** | 0.6082 | **0.9731** |

**Key Observations:**
* **Accuracy** is high for all models (>96%) but can be misleading due to class imbalance
* **Precision:** Random Forest has the best precision (93.10%), meaning when it predicts fraud, it's correct 93% of the time
* **Recall:** Logistic Regression catches the most frauds (89.44%), but with many false alarms
* **F1-Score:** Random Forest provides the best balance (0.8372)
* **ROC-AUC:** All models show excellent discrimination ability (>0.95)

### Business Impact Analysis

Assuming average fraud amount = $100, investigation cost = $5 per false positive, and missed fraud cost = $100:

| Model | Frauds Caught | Frauds Missed | False Alarms | Net Savings |
|-------|---------------|---------------|--------------|-------------|
| Logistic Regression | 127 ($12,700) | 15 ($1,500) | 2,672 ($13,360) | **-$2,160** |
| Random Forest | 108 ($10,800) | 34 ($3,400) | 8 ($40) | **$7,360** |
| Gradient Boosting | 118 ($11,800) | 24 ($2,400) | 128 ($640) | **$8,760** |

### Interpretation (The "Knowledge")

This is the most important part of KDD. By analyzing the `feature_importances_` from our Random Forest model, we extracted actionable knowledge:

**🏆 Top 5 Most Important Fraud Indicators:**

1. **`V14`** (Importance: 0.1443) - The most critical PCA-transformed feature for detecting fraud
2. **`V10`** (Importance: 0.1353) - Second most important anonymized feature
3. **`V4`** (Importance: 0.0956) - Another key PCA component
4. **`V12`** (Importance: 0.0944) - Fourth most discriminative feature
5. **`PCA_std`** (Importance: 0.0882) - Our engineered feature capturing variance across all PCA components

**Additional Key Insights:**

* **Feature Engineering Impact:** Our engineered features (`PCA_std`, `PCA_sum`, `PCA_mean`) rank among the top 10 most important features, validating our transformation phase
* **Time Patterns:** `TimePeriod_Evening` appears in top 15 features, suggesting fraudsters may prefer certain times of day
* **Interaction Effects:** `V1_V3_interaction` and `V1_V2_interaction` provide additional predictive power
* **Amount Less Critical:** Despite intuition, transaction amount-based features rank lower, indicating fraud occurs across all transaction sizes

### Model Recommendation

**🎯 RECOMMENDED MODEL: Random Forest**

This model provides the optimal balance for production deployment:
* **High Precision (93.10%):** Minimizes false alarms, reducing investigation costs
* **Good Recall (76.06%):** Catches 3 out of 4 fraud cases
* **Best F1-Score (0.8372):** Optimal harmonic mean of precision and recall
* **Strong Business Value:** Net savings of $7,360 compared to $8,760 for Gradient Boosting, but with 94% fewer false positives (8 vs. 128)
* **Interpretability:** Provides clear feature importance rankings for ongoing fraud pattern analysis

---

## Discovered Knowledge Summary

This KDD process revealed that:

1. **PCA features V14, V10, V4, and V12** are the strongest fraud indicators (likely capturing transaction behavior patterns)
2. **Variability in transaction patterns** (captured by `PCA_std`) is highly predictive of fraud
3. **Feature engineering significantly improves detection** - engineered features rank in top 10
4. **Class imbalance handling is critical** - SMOTE improved model performance dramatically
5. **Time-of-day matters** - Evening transactions show different fraud patterns
6. **Random Forest provides the best production model** with 93% precision and 76% recall

This discovered knowledge enables financial institutions to build effective real-time fraud detection systems that balance fraud prevention with customer experience.

---

## Repository Structure

```
kdd_fraud_detection_outputs/
├── kdd_fraud_detection_output.txt          # Complete text log
├── 01_class_distribution.png               # Class imbalance visualization
├── 02_feature_distributions.png            # Amount and Time distributions
├── 03_scaling_comparison.png               # Before/after scaling
├── 04_smote_comparison.png                 # SMOTE effect visualization
├── 05_confusion_matrices.png               # Model confusion matrices
├── 06_roc_curves.png                       # ROC curves comparison
├── 07_precision_recall_curves.png          # Precision-Recall curves
└── 08_feature_importance.png               # Top 15 important features
```

---

## How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place `creditcard.csv` in your working directory
3. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
   ```
4. Run the Python script (all cells sequentially)
5. Review outputs in `kdd_fraud_detection_outputs/` directory

---
