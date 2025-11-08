## A Step-by-Step Guide to Predicting Employee Attrition with CRISP-DM

The CRISP-DM (Cross-Industry Standard Process for Data Mining) model is a 6-phase cyclical process that provides a robust framework for managing data science projects.

**Project Focus:** Employee Attrition Prediction using IBM HR Analytics Dataset

### 1. Business Understanding

**The Problem:** Our project addresses a critical business challenge facing modern organizations. Our client, a mid-sized technology company called "TechCorp," has experienced a 25% increase in employee turnover over the last year. This is extremely costly, as replacing an employee can cost 50-200% of their annual salary when considering recruitment, training, lost productivity, and institutional knowledge loss.

**The Business Goal:** The company's objective is clear: **reduce employee attrition by at least 20%** in the next twelve months.

**Our Project's Role:** While some turnover is inevitable (retirement, relocation, career changes), we can target *preventable* attrition. Our project's objective is to build a predictive system that can **identify which *current* employees are at high risk of leaving** the organization within the next 90 days.

**Defining Success:**
* **Business Success:** The HR department successfully uses our tool to implement targeted retention strategies (e.g., career development plans, compensation adjustments, work-life balance improvements) that result in a measurable decrease in the attrition rate.
* **Data Mining Success:** We will build a predictive model (a binary classifier) that achieves at least **75% recall**. High **recall** (also called sensitivity) is our top priority, meaning we want to correctly identify as many of the *actual* potential leavers as possible, even if we accidentally flag some stable employees. This business-aligned metric ensures we don't miss at-risk talent.

### 2. Data Understanding

With our business goal established, we now need to acquire and examine the raw data. This phase is about understanding what we have *before* we start cleaning and modeling.

**Data Collection:**
For this project, we use the **IBM HR Analytics Employee Attrition Dataset**, a publicly available dataset from Kaggle/IBM that contains realistic employee data with demographics, job roles, satisfaction metrics, and attrition history.

**Data Description (Data Dictionary):**
Our dataset contains **1,470 rows** (one per employee) and the following **35 columns**:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `Age` | Integer | Employee's age in years (18-60). |
| `Attrition` | String | **(Our Target Variable)** "Yes" = Left company, "No" = Stayed. |
| `BusinessTravel` | String | Frequency of travel: "Non-Travel", "Travel_Rarely", "Travel_Frequently". |
| `DailyRate` | Integer | Daily salary rate in dollars. |
| `Department` | String | Employee's department: "Sales", "Research & Development", "Human Resources". |
| `DistanceFromHome` | Integer | Distance from home to office in miles. |
| `Education` | Integer | Education level (1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor). |
| `EducationField` | String | Field of education (e.g., "Life Sciences", "Medical", "Marketing"). |
| `EmployeeCount` | Integer | Constant value (1) - not useful for modeling. |
| `EmployeeNumber` | Integer | Unique employee identifier. |
| `EnvironmentSatisfaction` | Integer | Satisfaction with work environment (1=Low, 4=Very High). |
| `Gender` | String | "Male" or "Female". |
| `HourlyRate` | Integer | Hourly pay rate. |
| `JobInvolvement` | Integer | Level of job involvement (1=Low, 4=Very High). |
| `JobLevel` | Integer | Job level/seniority (1-5, higher is more senior). |
| `JobRole` | String | Specific role (e.g., "Sales Executive", "Research Scientist"). |
| `JobSatisfaction` | Integer | Overall job satisfaction (1=Low, 4=Very High). |
| `MaritalStatus` | String | "Single", "Married", or "Divorced". |
| `MonthlyIncome` | Integer | Monthly salary in dollars. |
| `MonthlyRate` | Integer | Monthly pay rate. |
| `NumCompaniesWorked` | Integer | Number of companies worked at previously. |
| `Over18` | String | Constant value ("Y") - all employees are over 18. |
| `OverTime` | String | Works overtime? "Yes" or "No". |
| `PercentSalaryHike` | Integer | Last salary increase percentage. |
| `PerformanceRating` | Integer | Latest performance rating (3=Excellent, 4=Outstanding). |
| `RelationshipSatisfaction` | Integer | Satisfaction with workplace relationships (1=Low, 4=Very High). |
| `StandardHours` | Integer | Constant value (80) - standard working hours. |
| `StockOptionLevel` | Integer | Stock option level (0-3). |
| `TotalWorkingYears` | Integer | Total years of professional experience. |
| `TrainingTimesLastYear` | Integer | Number of training sessions attended last year. |
| `WorkLifeBalance` | Integer | Work-life balance rating (1=Bad, 4=Best). |
| `YearsAtCompany` | Integer | Tenure at current company in years. |
| `YearsInCurrentRole` | Integer | Years in current role. |
| `YearsSinceLastPromotion` | Integer | Years since last promotion. |
| `YearsWithCurrManager` | Integer | Years with current manager. |

**Initial Data Exploration & Quality Report:**
After loading the data, we perform an initial exploratory data analysis (EDA) and find several key insights:

* **Missing Values:** ✅ **No missing values found!** This is a clean dataset, which is rare in real-world scenarios.
* **Data Types:** Mix of numerical (26 columns) and categorical (9 columns) features. Some numerical features are actually ordinal categories (e.g., Education, JobSatisfaction).
* **Constant Columns:** `EmployeeCount`, `StandardHours`, and `Over18` have only one unique value each—these provide zero information and must be dropped.
* **ID Column:** `EmployeeNumber` is a unique identifier and should be removed to prevent data leakage.
* **Target Variable Imbalance:** This is our most critical discovery. The dataset is highly **imbalanced**: 
  - **83.9%** of employees stayed (`Attrition = No`)
  - **16.1%** of employees left (`Attrition = Yes`)
  
  This severe imbalance (5.2:1 ratio) will make it extremely difficult for a naive model to learn what a "leaver" looks like. The model might just predict "No" for everyone and achieve 84% accuracy while being completely useless.

### 3. Data Preparation

This phase (also called "data wrangling" or "data munging") is where we execute the cleanup plan informed by our Data Understanding. This is often the most time-consuming step (up to 80% of a project) but is critical for building an accurate model.

Our goal is to create a final, clean "feature table." Here are the transformation "recipes" we used:

**1. Target Variable Encoding:**
* **`Attrition` ("Yes"/"No"):** Converted to binary numeric: `Yes = 1`, `No = 0`. This is required for all machine learning algorithms.

**2. Removing Non-Informative Columns:**
* **Dropped 4 columns:** `EmployeeCount`, `StandardHours`, `Over18` (constants), and `EmployeeNumber` (ID).
* **Rationale:** These columns have zero predictive power. Including them would just add noise and computational overhead.

**3. Encoding Categorical Variables:**
* **7 categorical columns** identified: `BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`, `OverTime`.
* **Method:** Used **Label Encoding** to convert text categories into numeric codes (e.g., "Sales" → 0, "R&D" → 1).
* **Note:** In production, one-hot encoding might be preferred for non-ordinal categories, but label encoding works well with tree-based models and keeps dimensionality lower.

**4. Feature Engineering (The "Secret Sauce"):**
This is where we leverage domain knowledge to create *new* features that provide stronger predictive signals:

* **`YearsPerPromotion`:** Calculated as `(YearsSinceLastPromotion + 1) / (YearsAtCompany + 1)`. 
  - **Insight:** Employees who are promoted frequently (low ratio) likely feel valued and are less likely to leave. Employees stuck without promotions (high ratio) may be flight risks.

* **`YearsWithManager`:** Calculated as `YearsAtCompany - YearsWithCurrManager`.
  - **Insight:** This captures how many managers an employee has had. Frequent manager changes might indicate instability or conflict.

* **`TotalSatisfaction`:** Average of `EnvironmentSatisfaction`, `JobSatisfaction`, and `RelationshipSatisfaction`.
  - **Insight:** A composite satisfaction score is often more predictive than individual satisfaction metrics. Low overall satisfaction is a red flag.

* **`WorkLifeBalance_Score`:** Calculated as `WorkLifeBalance × (5 - JobInvolvement)`.
  - **Insight:** Combines work-life balance and job involvement into a burnout risk indicator. High involvement with poor balance suggests overwork.

**5. Feature Scaling:**
* **Method:** Applied **StandardScaler** to all numeric features.
* **Why:** Ensures all features are on the same scale (mean=0, std=1). This is crucial for algorithms like Logistic Regression that are sensitive to feature magnitudes.
* **Note:** Scaling is applied *separately* to train and test sets to prevent data leakage.

**6. Handling Class Imbalance (Critical for Modeling):**
* **`Attrition` (84% "No," 16% "Yes"):** If we train on this raw data, the model will learn to predict "No" for everyone and achieve 84% accuracy while completely failing to identify leavers.
* **Solution:** Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to the training data only.
* **How SMOTE Works:** Intelligently generates synthetic examples of the minority class (leavers) by interpolating between existing minority samples.
* **Result:** Training set balanced to perfect 50/50 split (986 "No", 986 "Yes").
* **Important:** SMOTE is *only* applied to training data. The test set remains imbalanced to reflect real-world conditions.

After these steps, we have a clean, feature-rich, and properly balanced dataset ready for modeling with **34 features** and **1,470 samples**.

### 4. Modeling

Now that our data is prepared, we can begin the modeling phase. Our business problem is to predict a "Yes/No" outcome (Attrition/Stay), which is a **binary classification** task.

**1. Data Splitting:**
First, we split our entire clean dataset into two parts using stratified sampling:
* **Training Set (80% = 1,176 samples):** The model learns patterns of attrition from this data.
* **Test Set (20% = 294 samples):** This data is kept in a "lockbox" and is *only* used at the very end to evaluate the model's performance on completely unseen data. This prevents overfitting/"cheating."

```python
from sklearn.model_selection import train_test_split

X = df_processed.drop('Attrition', axis=1)  # All features
y = df_processed['Attrition']               # Only target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**2. Model Selection:**
We train **four different algorithms** to compare performance and select the best one:

* **Logistic Regression:** A simple, interpretable linear model. Fast to train and provides probability scores. Works well with scaled features.

* **Decision Tree:** A non-linear model that creates human-readable decision rules. Can capture complex interactions but prone to overfitting.

* **Random Forest:** An ensemble of 100 decision trees. More robust than single trees, handles non-linear patterns well, and provides feature importance rankings.

* **Gradient Boosting:** An advanced ensemble method that builds trees sequentially, each correcting the previous tree's errors. Often achieves the highest accuracy but slower to train.

**3. Training Process:**
Each model is trained on the **SMOTE-balanced training data** (986 "No", 986 "Yes") to ensure it learns to recognize both classes equally well.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
```

### 5. Evaluation

In the modeling phase, we built four models. Now, we must evaluate their performance on the unseen, **imbalanced** test set (247 "No", 47 "Yes") to see which one best meets our business goals.

**Key Evaluation Metrics:**
Our dataset is imbalanced (84% "No," 16% "Yes"). This means **Accuracy** is a misleading metric. (A model that just predicts "No" every time would be 84% accurate but 100% useless for our business goal).

We must use a **Confusion Matrix** to see the *types* of errors our model makes:

```
                Predicted
              Stays  Leaves
Actual Stays   TN     FP      (TN = True Negatives, FP = False Positives)
       Leaves  FN     TP      (FN = False Negatives, TP = True Positives)
```

Based on this, our key metrics are:
* **Precision:** Of all employees our model *predicted* would leave, what percentage *actually* left? (We don't want to waste HR resources on false alarms).
  - Formula: `TP / (TP + FP)`

* **Recall (Our Top Priority):** Of all employees who *actually* left, what percentage did our model successfully *identify*? (This aligns with our business goal: catch as many at-risk employees as possible).
  - Formula: `TP / (TP + FN)`

* **F1-Score:** The harmonic mean of Precision and Recall. A balanced metric that's good for imbalanced datasets.
  - Formula: `2 × (Precision × Recall) / (Precision + Recall)`

* **ROC-AUC Score:** Measures the model's ability to distinguish between classes across all threshold values. Range: 0.5 (random) to 1.0 (perfect).

**Model Performance Results:**
We evaluated all four models on the test set. Here are their scores:

| Model | Accuracy | ROC-AUC | **Recall (Attrition)** | Precision (Attrition) | F1-Score (Attrition) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **76.5%** | **0.808** | **🏆 76.6%** | 38.3% | 51.1% |
| Decision Tree | 77.9% | 0.601 | 42.6% | 34.5% | 38.1% |
| Random Forest | 84.7% | 0.796 | 27.7% | 54.2% | 36.6% |
| Gradient Boosting | 84.0% | 0.794 | 34.0% | 50.0% | 40.5% |

**Confusion Matrix Analysis (Logistic Regression):**
```
                Predicted
              Stays  Leaves
Actual Stays    189     58    (189 correctly identified as staying, 58 false alarms)
       Leaves    11     36    (36 correctly identified as leaving, 11 missed)
```

**Evaluation & Model Selection:**
The **Logistic Regression** is the clear winner for our business objectives:

* **Highest Recall (76.6%):** Successfully identifies **36 out of 47** employees who actually left. This means we catch **3 out of every 4** at-risk employees—excellent for our retention goal!

* **Acceptable Precision (38.3%):** Of the 94 employees flagged as high-risk (36 + 58), 36 actually left. While this means 58 false alarms, this is acceptable because:
  - Cost of missing a leaver (losing talent, replacement costs) >> Cost of false alarm (extra check-in meeting)
  - HR can prioritize the highest-risk scores within the flagged group

* **Strong ROC-AUC (0.808):** Indicates excellent discrimination ability. The model can reliably rank employees by attrition risk.

**Why Not Random Forest or Gradient Boosting?**
While they have higher accuracy (84-85%), they have **terrible recall** (28-34%). They correctly identify stable employees but miss most of the leavers—the exact opposite of what we need!

**Reviewing Business Goals:**
Let's check back with our goals from Phase 1:
* **Data Mining Success (Goal: ≥75% Recall):** We achieved **76.6% Recall**. ✅ **(Met)**
* **Business Success (Goal: Reduce attrition):** Our model provides HR with a highly actionable list of at-risk employees to target with retention strategies. ✅ **(Supported)**

The **Logistic Regression model is approved** for deployment. It is effective, interpretable, and directly aligned with the business objective.

### 6. Deployment

We have an evaluated, high-performing Logistic Regression model that is approved by the business. The final step is to integrate it into the company's HR operations.

**Deployment Plan:**
We propose a **quarterly batch-scoring system with monthly monitoring**.

1. **Model Packaging:** 
   - Save the trained Logistic Regression model and StandardScaler as serialized objects (`.pkl` files).
   - Create a production-ready Python script that loads the model and makes predictions on new data.

2. **Quarterly Scoring Schedule:** 
   A scheduled job will run on the 1st of every quarter (Jan/Apr/Jul/Oct) at 2 AM:
   - **Data Pull:** Extract latest employee data from HR database (all current employees).
   - **Data Preprocessing:** Apply the same transformations (feature engineering, encoding, scaling).
   - **Risk Scoring:** Use the Logistic Regression model to generate an "Attrition Risk Score" (0.0 to 1.0 probability) for every employee.
   - **Database Update:** Save scores to a new `employee_risk_scores` table with fields:
     - `EmployeeID`, `RiskScore`, `RiskCategory` (Low/Medium/High), `ScoringDate`

3. **HR Dashboard Integration:**
   - Build a web dashboard (using Tableau/Power BI) connected to the risk scores database.
   - **Features:**
     - Interactive table showing all employees sorted by risk score (highest first)
     - Filters by department, manager, job role, tenure
     - Risk distribution visualizations (pie charts, histograms)
     - Historical tracking: Compare quarterly risk scores to identify trends
   
4. **Actionable Interventions:**
   HR managers receive automated alerts and can take action:
   - **High Risk (Score > 0.7):** Immediate one-on-one meeting to discuss career goals, concerns, and retention strategies (promotion track, salary review, flex work).
   - **Medium Risk (Score 0.4-0.7):** Quarterly check-ins, career development planning, training opportunities.
   - **Low Risk (Score < 0.4):** Standard annual reviews, engagement surveys.

5. **Email Automation:**
   - Weekly digest sent to department heads listing their top 5 highest-risk employees.
   - Monthly executive summary to CHRO showing company-wide attrition risk trends.

**Monitoring & Maintenance (The "Cycle"):**
Deployment is *not* the end. The CRISP-DM model is a cycle for a reason:

* **Performance Monitoring:** 
  - Track actual attrition vs. predicted attrition each quarter.
  - Calculate real-world recall: Of employees who left, what % were flagged as high-risk?
  - Target: Maintain ≥70% recall in production.

* **Concept Drift Detection:** 
  - Employee behavior changes over time (e.g., post-pandemic remote work preferences).
  - Monitor for distribution shifts in key features (e.g., average satisfaction scores dropping).
  - Set alerts if actual attrition rate deviates >5% from predictions.

* **Quarterly Model Retraining:** 
  - Retrain the model every quarter using the most recent 2 years of employee data.
  - Include newly collected attrition outcomes to capture evolving patterns.
  - A/B test: Compare old model vs. new model on validation set before deploying.
  - This ensures the model stays "fresh" and adapts to organizational changes, starting the CRISP-DM cycle all over again.

* **Continuous Improvement:**
  - Collect feedback from HR managers on false positives/negatives.
  - Experiment with new features (e.g., recent performance review scores, participation in employee engagement programs).
  - Test alternative algorithms (e.g., XGBoost, Neural Networks) if business needs change.

**Expected Business Impact:**
Based on the model's 76.6% recall and assuming HR successfully retains 50% of identified high-risk employees:
- **Baseline:** 16% annual attrition rate
- **With Model:** Estimated reduction to ~10% annual attrition
- **Cost Savings:** For a 1,000-employee company, preventing ~30 departures/year could save $1.5M-$6M annually in replacement costs.

---

## Project Summary

**Dataset:** IBM HR Analytics Employee Attrition Dataset (1,470 employees, 35 features)

**Business Objective:** Predict employee attrition to enable proactive retention strategies

**Methodology:** CRISP-DM (Cross-Industry Standard Process for Data Mining)

**Best Model:** Logistic Regression
- **Recall:** 76.6% (catches 3 out of 4 leavers)
- **ROC-AUC:** 0.808
- **Deployment Status:** Production-ready

**Key Success Factors:**
1. Strong business alignment (prioritizing recall over accuracy)
2. Effective handling of class imbalance with SMOTE
3. Domain-driven feature engineering
4. Comprehensive evaluation on business-relevant metrics

**Next Steps:**
1. Deploy to production HR system
2. Integrate with HR dashboard
3. Monitor real-world performance
4. Retrain quarterly with new data

---

## Files Generated

**Code:** `employee-attrition-with-logging.py`

**Output Logs:** `CRISP_DM_Output_[timestamp].txt`

**Visualizations:**
- `attrition_distribution_[timestamp].png` - Target variable distribution
- `confusion_matrix_Logistic_Regression_[timestamp].png` - Best model performance
- `confusion_matrix_Decision_Tree_[timestamp].png`
- `confusion_matrix_Random_Forest_[timestamp].png`
- `confusion_matrix_Gradient_Boosting_[timestamp].png`
- `model_comparison_[timestamp].png` - Side-by-side model comparison

---

*This project demonstrates the complete CRISP-DM lifecycle from business problem to production-ready solution, with a focus on actionable insights and real-world deployment considerations.*
