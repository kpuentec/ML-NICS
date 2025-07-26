from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, scaler, model_path='models/nids_random_forest.pkl', scaler_path='models/scaler.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)