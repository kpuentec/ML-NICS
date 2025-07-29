from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from joblib import dump

from src.preprocess import load_data, preprocess, scale_features
from src.evaluate import evaluate_model
import os
import sys

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")

def train_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df_train, df_test = load_data()
    X_train, y_train = preprocess(df_train)
    X_test, y_test = preprocess(df_test)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Save scaler
    dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    # Logistic Regression
    lr = LogisticRegression(solver='lbfgs', max_iter=2000, class_weight='balanced')
    lr.fit(X_train_scaled, y_train)
    print("=== Logistic Regression Evaluation (default 0.5) ===")
    evaluate_model(lr, X_test_scaled, y_test, threshold=0.5)

    print("=== Logistic Regression Evaluation (best F1) ===")
    evaluate_model(lr, X_test_scaled, y_test, threshold=0.035)

    dump(lr, os.path.join(MODELS_DIR, "logistic_regression.pkl"))

    # SVM
    svm = SVC(kernel='rbf', probability=True, class_weight='balanced', C=1.0, gamma='scale')
    svm.fit(X_train_scaled, y_train)
    print("\n=== SVM (RBF Kernel) Evaluation ===")
    evaluate_model(svm, X_test_scaled, y_test)
    dump(svm, os.path.join(MODELS_DIR, "svm_rbf.pkl"))

    print("\nModels and scaler saved in /models directory.")

if __name__ == "__main__":
    train_models()
    sys.exit(0)