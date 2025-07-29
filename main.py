import os
import pandas as pd
import joblib

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")

def list_models():
    if not os.path.exists(MODELS_DIR):
        print("No models directory found.")
        return []
    models = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    if not models:
        print("No models found. Please run train.py first.")
    return models

def load_model(model_name):
    path = os.path.join(MODELS_DIR, model_name)
    return joblib.load(path)

def predict_single(model, scaler, input_data):
    df = pd.DataFrame([input_data])
    for col in ['protocol_type', 'service', 'flag']:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0, 1]
    return prediction, prob

def main():
    print("=== Network Intrusion Detection System (NIDS) ===")
    
    models = list_models()
    if not models:
        return
    print("\nAvailable Models:")
    for i, m in enumerate(models, 1):
        print(f"{i}. {m}")

    choice = int(input("\nChoose a model (number): "))
    chosen_model = models[choice - 1]
    print(f"\nLoading model: {chosen_model} ...")
    model = load_model(chosen_model)

    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        print("Scaler not found. Did you run train.py?")
        return
    scaler = joblib.load(scaler_path)

    print("\nEnter network traffic features (example: duration, protocol_type, service, flag, src_bytes, dst_bytes...)")
    input_data = {
        "duration": int(input("Duration: ")),
        "protocol_type": input("Protocol (e.g., tcp/udp/icmp): "),
        "service": input("Service (e.g., http/private/ftp): "),
        "flag": input("Flag (e.g., SF/S0/REJ): "),
        "src_bytes": int(input("Source Bytes: ")),
        "dst_bytes": int(input("Destination Bytes: "))
    }

    prediction, prob = predict_single(model, scaler, input_data)

    print("\n=== Prediction Result ===")
    print(f"Prediction: {'Attack' if prediction == 'attack' else 'Normal'}")
    print(f"Probability of Attack: {prob:.4f}")

if __name__ == "__main__":
    main()
