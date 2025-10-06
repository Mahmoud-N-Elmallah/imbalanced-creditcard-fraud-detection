# Credit Card Fraud Detection using Machine Learning

This project focuses on detecting fraudulent credit card transactions in a **highly imbalanced dataset** using advanced data preprocessing, sampling, and model tuning techniques.  
The workflow demonstrates a full applied machine learning pipeline — from data cleaning and feature engineering to model evaluation.

---

##  Dataset
The dataset used is `creditcard.csv` that can be found here "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" , which contains anonymized PCAed transaction data and a binary target column `Class`:
- `0`: Legitimate transaction  
- `1`: Fraudulent transaction  
- The fraud class represents only **0.166%** of the total samples, making this a **severely imbalanced classification problem**.

---

##  Data Preprocessing and Exploration
- Checked for **missing values** and **duplicates**, then removed duplicates.
- Explored data distributions using `seaborn` and `matplotlib` for both individual features and correlations with the target.
- Noted that:
  - `Amount` was **positively skewed**.
  - `Time` needed **scaling**.
- Applied **log transformation** to `Amount` and `Time`.
- Created new features:
  - `amount_per_unit_time` = `Amount / Time`
  - `10_rolls_amount_mean` = rolling mean of the last 10 transaction amounts
- Applied **KMeans clustering** (`n_clusters=10`) on the scaled features to add a new categorical feature `clusterd`, representing the assigned cluster label.

---

##  Feature Scaling
Used **RobustScaler** to normalize features while minimizing the effect of outliers.

---

##  Handling Imbalanced Data
Tested multiple **oversampling** and **undersampling** strategies from `imblearn`:
- **Oversampling**: `SMOTE`, `BorderlineSMOTE`, `ADASYN`, `SMOTEN`, `SMOTEENN`, `SMOTETomek`, `KMeansSMOTE`
- **Undersampling**: `AllKNN`, `RandomUnderSampler`, `ClusterCentroids`

The best-performing sampler was **SMOTEN**, which was selected for the final pipeline.

---

##  Model Selection
The following classifiers were compared under the same SMOTEN oversampling setup:
- `LogisticRegression`
- `DecisionTreeClassifier`
- `AdaBoostClassifier`
- `GradientBoostingClassifier`
- `RandomForestClassifier`
- `SVC`
- `XGBClassifier`
- `LGBMClassifier`

Both **Random Forest** and **XGBoost** achieved strong results, but **XGBoost** was chosen due to ist training efficiency and ease of tuning on large datasets.

---

##  Hyperparameter Tuning
- Used **RandomizedSearchCV** with **StratifiedKFold (10 splits)** for robust cross-validation.
- Parameter distribution (`xgb_param_dist`) included:
  - `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`, and `scale_pos_weight`.
- The model was wrappped in an `imblearn` **Pipeline** combining:
  1. `SMOTEN` oversampling  
  2. `XGBClassifier`

---

##  Train/Test Split
- Used a **70/30** train–test split (`stratify=y`, `random_state=42`).
- Cross-validation within RandomizedSearchCV handled the internal validation.

---

##  Results

**Training Data:**
| Metric | Value |
|--------|--------|
| Precision (fraud) | 0.99 |
| Recall (fraud) | 1.00 |
| F1 (fraud) | 0.99 |
| Macro F1 | 1.00 |

**Test Data (Unseen):**
| Metric | Value |
|--------|--------|
| Precision (fraud) | 0.93 |
| Recall (fraud) | 0.77 |
| F1 (fraud) | 0.85 |
| Macro F1 | 0.92 |

The model achieves a **macro F1-score of 0.92 on unseen data**, showing strong generalization adn effective handling of class imbalance.  
The performance gap between training and test sets is realistic, indicating **no major overfitting**.

---

##  Evaluation
- `classification_report` for detailed metrics.
- `ConfusionMatrixDisplay` and `RocCurveDisplay` to visualize performance.
- `f1_macro` used as the primary scoring metric.

---

##  Key Techniques and Skills Demonstrated
- Data cleaning and feature engineering
- Handling extreme class imbalance using undersampling and oversampling techniques
- Feature scaling 
- Clustering with **KMeans** for feature enrichment
- Cross-validation with **StratifiedKFold**
- Model selection across multiple ML algorithms
- Hyperparameter optimization using **RandomizedSearchCV**
- Evaluation with ROC curve, confusion matrix, and F1 metrics
- Integration of **imblearn pipelines** for balanced model training

---

##  Libraries Used
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`
- `imblearn`
- `xgboost`, `lightgbm`, `catboost`
- `scipy`

---

##  Final Remarks
> The model performs strongly on an imbalanced dataset, achieving high precision and recall for the minority class.  
> This project demonstrates end-to-end data science workflow — from exploration and feature engineering to sampling, model tuning, and evaluation.

---

