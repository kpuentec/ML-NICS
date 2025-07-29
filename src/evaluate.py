from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve
)
import numpy as np

def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_bin = y_test.map({'normal': 0, 'attack': 1})

    attack_index = list(model.classes_).index('attack')
    y_score = model.predict_proba(X_test)[:, attack_index]
    y_pred = np.where(y_score >= threshold, 'attack', 'normal')

    print(f"\n=== Evaluation at Threshold = {threshold:.2f} ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_bin, y_score))

    prec, rec, thr = precision_recall_curve(y_bin, y_score)
    best_idx = np.argmax(2 * prec * rec / (prec + rec + 1e-6))  # F1 max
    print(f"\nBest threshold (by F1): {thr[best_idx]:.3f}")
    print(f"Precision at best: {prec[best_idx]:.3f}")
    print(f"Recall at best: {rec[best_idx]:.3f}")
