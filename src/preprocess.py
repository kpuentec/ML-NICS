import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")

def load_data():
    fields_path = os.path.join(DATA_DIR, "FieldNames.csv")
    train_path = os.path.join(DATA_DIR, "KDDTrain+.txt")
    test_path = os.path.join(DATA_DIR, "KDDTest+.txt")

    df_fields = pd.read_csv(fields_path, header=None, names=["feature", "type"])
    col_names = df_fields["feature"].tolist()

    col_names += ["class", "difficulty"]

    df_train = pd.read_csv(train_path, names=col_names)
    df_test = pd.read_csv(test_path, names=col_names)

    return df_train, df_test


def preprocess(df):
    df = df.copy()

    if "difficulty" in df.columns:
        df.drop("difficulty", axis=1, inplace=True)

    for col in ["protocol_type", "service", "flag"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    df["class"] = df["class"].apply(lambda x: "normal" if x == "normal" else "attack")

    X = df.drop("class", axis=1)
    y = df["class"]
    return X, y

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
