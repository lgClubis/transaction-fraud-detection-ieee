# Reflection

## 1. Framing the Problem Beyond Classification

A key learning from this project is that fraud detection should not be framed purely as a binary classification problem.  
In real-world settings, decisions are constrained by operational capacity, customer experience, and financial risk.

Instead of asking whether a transaction is fraudulent, the more relevant question is how transactions should be **ranked by risk** to support downstream decisions such as manual review or automated intervention.

This shift in perspective strongly influenced both the evaluation strategy and the choice of metrics used in this project.

---

## 2. Why ROC-AUC Alone Is Insufficient

While ROC-AUC is a useful indicator of overall ranking quality, it does not capture the operational usefulness of a fraud model.

In highly imbalanced settings, a model with a strong AUC can still perform poorly when only a small fraction of transactions can be reviewed.  
Therefore, this project emphasized **Recall@Top-k%** and **Precision@Top-k%**, which directly reflect real-world constraints.

This reinforced the importance of aligning evaluation metrics with business objectives rather than relying on abstract performance measures.

---

## 3. Importance of Time-Aware Validation

One of the most important insights from this project was the risk of overly optimistic evaluation when using random train-test splits.

Fraud patterns evolve over time, and future transactions are not independent of past behavior.  
Time-based cross-validation was used to simulate realistic deployment scenarios, ensuring that the model only learned from historical data when predicting future events.

The results demonstrated that while performance fluctuates across time periods, the model remained stable without catastrophic degradation, which is critical for practical deployment.

---

## 4. Feature Engineering and Target Leakage Awareness

Target encoding for categorical features can significantly improve performance, but it introduces a high risk of data leakage if not handled carefully.

In this project, target encoding was applied in a strictly time-aware manner, ensuring that validation data never influenced the encoding statistics.  
Interestingly, the performance gains from target encoding were marginal, highlighting that not all theoretically beneficial techniques lead to improvements in every dataset.

This emphasized the importance of empirical validation over assumptions.

---

## 5. Limitations and Production Considerations

Although the model performed well under offline evaluation, several limitations remain.

The project does not include cost-sensitive optimization, adaptive retraining schedules, or real-time feedback mechanisms.  
Additionally, entirely novel fraud patterns may initially be treated as neutral due to limited historical information.

In a production environment, this model would need to be complemented by monitoring systems, anomaly detection, and periodic retraining to manage concept drift and evolving fraud strategies.

---

## 6. Personal Takeaway

This project strengthened my understanding that effective business analytics is not about maximizing a single metric, but about making informed, responsible decisions under uncertainty.

The combination of technical modeling, evaluation design, and critical reflection provided a holistic perspective on how analytical models support real-world decision-making.
