# transaction-fraud-detection-ieee
# Transaction Fraud Detection  
### A Time-Aware, Business-Oriented Machine Learning Approach
See [Reflection.md](REFLECTION.md) for methodological insights and limitations.



## 1. Business Problem

Credit card fraud causes significant financial losses and customer dissatisfaction.  
In practice, fraud detection systems must operate under **limited review capacity**, meaning that only a small fraction of transactions can be manually inspected or automatically blocked.

The goal of this project is **not only to build an accurate classifier**, but to develop a **risk-ranking model** that prioritizes the most suspicious transactions under realistic operational constraints.

---

## 2. Dataset

- **IEEE-CIS Fraud Detection Dataset** (Kaggle)
- ~590,000 transactions
- Highly imbalanced target variable  
  - Fraud rate ≈ **3.5%**
- Contains strong **temporal dependencies** and **concept drift**
- Mixture of numerical and categorical features

---

## 3. Methodological Approach

### 3.1 Data Preparation
- Transaction and identity tables are merged
- Data is **sorted chronologically** using `TransactionDT`
- Processed dataset is stored in **Parquet format** for efficiency and reproducibility

### 3.2 Modeling
- Gradient boosting classifier using **LightGBM**
- Numeric features as baseline
- Target Encoding for selected categorical features (leakage-free, time-aware)
- No random train-test split

### 3.3 Evaluation Strategy (Key Design Choice)
Traditional metrics such as ROC-AUC alone are insufficient in fraud detection.

Therefore, the model is evaluated using:
- **Time-based validation** (no random cross-validation)
- **Business-oriented metrics**:
  - Recall@Top-k%
  - Precision@Top-k%

This simulates real-world deployment conditions where only a limited percentage of transactions can be reviewed.

---

## 4. Results

### 4.1 Single Time Split (80/20)
- ROC-AUC ≈ **0.915**
- Precision@1% consistently **> 90%**
- Recall@1% ≈ **25–30%**

This means that with only 1% of transactions reviewed, the model identifies roughly **one quarter of all fraud cases** while maintaining a very clean review queue.

---

### 4.2 Time-Based Cross-Validation (Stability Analysis)

Four chronological folds were evaluated to assess temporal robustness.

| Fold | ROC-AUC | Recall@1% | Precision@1% |
|------|---------|-----------|--------------|
| 1    | ~0.90   | ~0.26     | ~0.93        |
| 2    | ~0.93   | ~0.22     | ~0.94        |
| 3    | ~0.93   | ~0.27     | ~0.95        |
| 4    | ~0.92   | ~0.29     | ~0.90        |

**Key observation:**  
Performance remains stable over time with no catastrophic degradation, indicating robustness to concept drift.

---

## 5. Why Time-Based Validation Matters

Random train-test splits can lead to overly optimistic performance estimates in fraud detection.

This project explicitly uses **time-aware validation**, ensuring that:
- The model only learns from past data
- Future transactions are treated as truly unseen
- New devices or patterns are handled conservatively

This approach reflects real-world fraud system deployment.

---

## 6. Project Structure

src/
    make_dataset.py # Data preparation
    train_lgbm.py # Model training
    evaluate_lgbm.py # Business-focused evaluation
    time_cv_lgbm.py # Time-based cross-validation
runs/
    *.json # Stored evaluation results
models/
    *.txt # Trained model artifacts
data/
    raw/ # Original CSV files (not tracked)
    processed/ # Parquet dataset

---

## 7. Limitations & Future Work

- No cost-sensitive optimization (fraud vs. customer friction)
- No real-time feedback loop
- No automated retraining schedule
- No anomaly detection for entirely new fraud patterns

These aspects would be required for a full production system.

---

## 8. Key Takeaways

This project demonstrates:
- Business-oriented machine learning thinking
- Proper handling of imbalanced data
- Responsible model validation under temporal constraints
- Awareness of model risk, drift, and operational trade-offs

The focus is on **decision support quality**, not leaderboard optimization.
