# Project: Classifying Wine Quality with the SEMMA Workflow

This project uses the **SEMMA (Sample, Explore, Modify, Model, Assess)** framework to build a multi-class classification model. The goal is to follow this technical workflow to accurately classify wines into quality categories (Poor, Good, Excellent) based on their physicochemical properties.

## Dataset Information

- **Source**: UCI Machine Learning Repository - Wine Quality Dataset
- **Total Records**: 6,497 wines (combination of red and white wines)
- **Features**: 12 physicochemical properties
- **Target Variable**: Quality categories derived from expert ratings

## Project Structure

```
wine_quality_output_YYYYMMDD_HHMMSS/
├── output_log.txt                    # Complete analysis log
├── 01_target_distribution.png        # Target variable visualization
├── 02_feature_distributions.png      # Feature distribution histograms
├── 03_confusion_matrix.png          # Best model confusion matrix
└── 04_model_comparison.png          # Model performance comparison
```

---

## SEMMA Workflow Implementation

### 1. Sample

The SEMMA process begins with **sampling** the data. The goal is to pull a representative, manageable subset of data from a large repository.

* **Our Data:** We are using the "Wine Quality Dataset," which contains 6,497 records—one for each wine sample (both red and white wines combined).
* **Our Sampling Strategy:** The dataset is already a manageable size. Therefore, our primary sampling step is to partition the full dataset into three distinct samples:
    * **Training Sample (70%):** The primary subset used to teach our models. *(4,547 samples)*
    * **Validation Sample (15%):** A subset used to tune model parameters and monitor training performance. *(975 samples)*
    * **Test Sample (15%):** A final, "lockbox" subset used *only once* at the end to **Assess** the chosen model's real-world performance. *(975 samples)*

**Key Implementation:**
- Used stratified sampling to maintain class distribution across all three sets
- Ensured reproducibility with `random_state=42`

---

### 2. Explore

With our samples defined, we move to the **explore** phase. This step is about visualizing the data to understand its properties, find relationships, and identify any problems.

* **Target Variable:** We first explore our target, `quality_category`. We find that it is a **multi-class** variable with 3 categories: Poor (≤5), Good (6-7), and Excellent (8-9). The class distribution reveals significant **imbalance**:
    - Good: 60.3% (2,740 samples)
    - Poor: 36.7% (1,668 samples)
    - Excellent: 3.1% (139 samples)

* **Feature Exploration:** We explore the 12 physicochemical features including:
    - Fixed acidity, volatile acidity, citric acid
    - Residual sugar, chlorides
    - Free sulfur dioxide, total sulfur dioxide
    - Density, pH, sulphates, alcohol content
    - Wine type (red/white)

* **Correlation Analysis:** Key discovery shows that **alcohol content** has the strongest positive correlation with wine quality (0.449), while **wine type** shows a negative correlation (-0.115), indicating red wines tend to have slightly different quality distributions.

* **Feature Scales:** We observe that feature scales vary dramatically:
    - Total sulfur dioxide ranges from 6 to 440
    - Density ranges from 0.987 to 1.039
    - Alcohol ranges from 8.0% to 14.9%

* **Key Discovery:** This difference in scales is a critical discovery. It tells us that our "Modify" step **must** include feature scaling, as models like K-Nearest Neighbors (KNN) or Support Vector Machines will fail if one feature's scale dominates all the others.

---

### 3. Modify

The **modify** phase is where we clean, transform, and prepare the data for modeling, based on our "Explore" discoveries.

* **Encoding the Target:** Our target variable `quality_category` is text (e.g., "Poor", "Good", "Excellent"). We use a **Label Encoder** to "modify" this text into numbers (0, 1, 2) that a machine learning algorithm can understand.

* **Handling Missing Data:** We check the dataset for missing values. We find it is **100% complete**, so no imputation is needed. This is a very clean dataset with no null values across any of the 6,497 records.

* **Feature Scaling (Our Main Task):** Based on our "Explore" discovery, this is our most important "Modify" step. We apply a **StandardScaler** to all 12 numerical features. This transformation rescales all features to have:
    - Mean = 0.0000
    - Standard deviation = 1.0000
    
    This ensures all features are weighted equally by the models and prevents any single feature from dominating the learning process.

**Important Note:** The scaler is **fit only on the training data** and then used to transform all three sets (train, validation, test). This prevents data leakage and ensures realistic performance evaluation.

---

### 4. Model

Now we **model** the data. In this phase, we apply various algorithms to our modified training sample to see which one can best learn the patterns that differentiate the 3 quality categories.

* **Model 1: K-Nearest Neighbors (KNN, k=7):** A simple "distance-based" model. We choose this specifically because its performance will prove that our "Modify" (scaling) step was necessary. KNN is highly sensitive to feature scales.
    - Validation Accuracy: 73.85%

* **Model 2: Support Vector Machine (SVM with RBF kernel):** A powerful non-linear model that finds optimal decision boundaries in high-dimensional space. Well-suited for complex classification problems.
    - Validation Accuracy: 76.72%

* **Model 3: Gradient Boosting Classifier:** An advanced "ensemble" model that builds multiple decision trees sequentially, with each tree correcting errors from previous ones. Known for high accuracy and robustness.
    - Validation Accuracy: 76.92%

* **Training:** All three models are trained on our 70% "Training Sample" (4,547 samples) using the scaled features. We monitor performance on the validation set to detect any overfitting.

---

### 5. Assess

The final step is to **assess** the trained models. We use our 15% "Test Sample" (the "lockbox" data with 975 samples) to get an honest, final evaluation of our best-performing model.

* **Assessment Metrics:** Since this is a multi-class problem with imbalanced classes, we use:
    - **Accuracy**: Overall percentage of correct predictions
    - **F1-Score (Weighted)**: Harmonic mean of precision and recall, weighted by class support. This metric accounts for the class imbalance we found in the "Explore" step.

* **Results:** We compare both accuracy and F1-scores of all three models on the test set:

| Model | Test Accuracy | F1-Score (Weighted) |
|-------|--------------|---------------------|
| K-Nearest Neighbors | 70.56% | 0.6983 |
| Support Vector Machine | 74.97% | 0.7358 |
| **Gradient Boosting** | **75.59%** | **0.7486** |

* **Final Assessment:** The **Gradient Boosting Classifier** is chosen as the best model with an F1-score of 0.7486 and accuracy of 75.59%. 

* **Confusion Matrix Analysis:** We generate a final confusion matrix to visualize its performance:
    - **Excellent wines (30 samples)**: Very difficult to predict (only 17% recall). The model tends to classify them as "Good" wines. This is expected due to the severe class imbalance (only 3% of training data).
    - **Good wines (587 samples)**: Excellent performance with 83% recall and 78% precision. The model excels at identifying this majority class.
    - **Poor wines (358 samples)**: Good performance with 68% recall and 71% precision. Some confusion with "Good" wines at the boundary.

* **Class-Specific Performance:**
    ```
    Excellent: Precision=0.83, Recall=0.17, F1=0.28
    Good:      Precision=0.78, Recall=0.83, F1=0.80
    Poor:      Precision=0.71, Recall=0.68, F1=0.70
    ```

This final assessment provides a complete picture of our model's strengths and weaknesses. The model performs well overall, with particular strength in identifying good-quality wines, but struggles with the rare "Excellent" category due to insufficient training examples.

---

## Key Insights and Conclusions

1. **Feature Importance**: Alcohol content emerged as the most important predictor of wine quality, followed by volatile acidity and citric acid.

2. **Class Imbalance Challenge**: The severe underrepresentation of "Excellent" wines (3.1% of data) significantly limits the model's ability to recognize this category. Future work could involve:
   - Oversampling excellent wines
   - Using SMOTE (Synthetic Minority Over-sampling Technique)
   - Collecting more high-quality wine samples

3. **Model Selection**: Gradient Boosting outperformed simpler models, demonstrating the value of ensemble methods for this complex classification task.

4. **Scaling Impact**: The performance difference between KNN (70.56%) and more sophisticated models (75.59%) validates the importance of proper feature scaling in the SEMMA workflow.

5. **Production Readiness**: With 75.59% accuracy, this model could be deployed as a decision support tool for wine quality assessment, particularly effective for distinguishing "Good" and "Poor" quality wines.

---

## Requirements

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Usage

Simply run the complete Python script. The code will:
1. Automatically download the dataset from UCI repository
2. Execute all SEMMA steps sequentially
3. Generate a timestamped output directory containing:
   - Complete text log of all outputs
   - High-resolution visualization images (300 DPI)
   - Summary report

## Author Notes

This project demonstrates the practical application of the SEMMA methodology for real-world classification problems. The structured approach ensures reproducibility, proper data handling, and rigorous model evaluation—all critical components of professional data science workflows.
