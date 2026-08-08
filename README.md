# Framingham Heart Study, 10-Year CHD Risk Prediction (Classification Assignment)

## Overview
This project builds a complete machine learning classification pipeline to predict
**`TenYearCHD`**, whether a patient will develop Coronary Heart Disease within 10 years,
using demographic, behavioral, and clinical data from the Framingham Heart Study
(`framingham.csv`, 4,238 rows, 16 columns).

## Files
| File | Description |
|---|---|
| `SMIT Framingham_Classification_Assignment - MFQ.ipynb` | Complete, fully-executed Jupyter Notebook with code, comments, and output (charts, tables, metrics) |
| `framingham.csv` | Dataset used |
| `README.md` | This file |

## Dataset
Features: `male`, `age`, `education`, `currentSmoker`, `cigsPerDay`, `BPMeds`,
`prevalentStroke`, `prevalentHyp`, `diabetes`, `totChol`, `sysBP`, `diaBP`, `BMI`,
`heartRate`, `glucose`.
Target: `TenYearCHD` (0 = did not develop CHD, 1 = developed CHD within 10 years).

## Workflow / Steps Performed
1. **Exploratory Data Analysis** — shape, dtypes, summary stats, missing-value check,
   target distribution, correlation heatmap, feature distributions by target class.
2. **Missing Value Handling**, `glucose` (~9%), `education` (~2.5%), and smaller amounts
   in `BPMeds`, `totChol`, `cigsPerDay`, `BMI`, `heartRate` were **median-imputed**
   (fit on training data only, applied to train & test, no data leakage).
3. **Encoding** — checked column dtypes; every column was already numeric
   (binary flags and an ordinal `education` scale), so **no additional encoding** was needed.
4. **Train/Test Split**, 80/20 stratified split performed **before** any imputation/
   scaling/balancing.
5. **Feature Scaling**, `StandardScaler` applied to all features (fit on train only).
6. **Balancing the Dataset** — target was significantly imbalanced (~85% No CHD vs ~15%
   CHD). **SMOTE** was applied to the training data only, creating a balanced 50/50
   training set; the test set was left untouched/imbalanced to reflect real-world conditions.
7. **Model Training & Comparison**, 8 classification models trained and evaluated:
   - Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Naive Bayes,
     Decision Tree, Random Forest, Gradient Boosting, XGBoost
8. **Metrics**, Accuracy, Precision, Recall, F1-Score, and ROC-AUC computed for every
   model on the held-out test set, plus full Classification Report and Confusion Matrix.
9. **Best Model Selection**, the model with the highest **F1-Score** was selected.
10. **Hyperparameter Tuning**, `GridSearchCV` (cross-validated) automatically tunes
    **whichever model actually won** step 9, using a parameter grid tailored to that
    model type (the notebook is written to adapt to any winning model, not hardcoded to one).
11. **Final Evaluation**, Classification Report, Confusion Matrix, ROC Curve, and
    Before-vs-After Tuning comparison for the final model, plus feature importance/coefficients.

## Results Summary (test set, before tuning)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression (best baseline)** | 0.665 | 0.249 | **0.597** | **0.352** | 0.697 |
| K-Nearest Neighbors | 0.643 | 0.212 | 0.496 | 0.297 | 0.599 |
| Support Vector Machine | 0.697 | 0.240 | 0.457 | 0.315 | 0.650 |
| Naive Bayes | 0.774 | 0.240 | 0.225 | 0.232 | 0.683 |
| Decision Tree | 0.724 | 0.213 | 0.302 | 0.250 | 0.551 |
| Random Forest | 0.793 | 0.230 | 0.155 | 0.185 | 0.647 |
| Gradient Boosting | 0.758 | 0.260 | 0.318 | 0.286 | 0.655 |
| XGBoost | 0.804 | 0.224 | 0.116 | 0.153 | 0.590 |

**Best Model: Logistic Regression**, it had the highest F1-Score (0.352) *and* by far the
best Recall (0.60), which matters most for a medical screening task where missing an
at-risk patient is far more costly than a false alarm. The tree-based models had higher
raw accuracy, but this was misleading: they mostly just predicted the majority "No CHD"
class, giving them very poor Recall (as low as 0.12–0.16), a classic imbalanced-data
pitfall that a plain "accuracy" metric hides.

### Hyperparameter Tuning Result
`GridSearchCV` searched over `C`, `penalty`, and `class_weight` for Logistic Regression.
The best parameters found (`C=0.1`, default `class_weight`) performed almost identically
to the untuned default model (F1 0.350 vs 0.352 baseline), telling us the default
settings were already close to optimal for this model/dataset combination. This is a
realistic and common outcome of hyperparameter tuning, not every tuning run yields a
dramatic improvement.

### Key Risk Factors (from Logistic Regression coefficients)
`age`, `cigsPerDay`, `male` (sex), and `sysBP` (systolic blood pressure) had the largest
positive coefficients, higher values of these push the prediction toward developing
CHD, consistent with established cardiovascular risk factors in medical literature.

## How to Run
1. Ensure the CSV file is in the same folder as the notebook.
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost jupyter
   ```
3. Open and run all cells:
   ```bash
   jupyter notebook "SMIT Framingham_Classification_Assignment - MFQ.ipynb"
   ```

## Notes / Possible Further Improvements
- Try `RandomizedSearchCV` or Bayesian optimization (Optuna) for a broader hyperparameter search.
- Try threshold-tuning instead of the default 0.5 cutoff, to further optimize the
  Recall/Precision tradeoff for a clinical screening use-case.
- Try alternate imbalance-handling techniques (`class_weight='balanced'`, ADASYN, or a
  combination of over/under-sampling).
- Try stacking/ensembling the top 2–3 models.
- Collect additional longitudinal or lifestyle data (diet, exercise, family history) to
  potentially improve predictive power.
