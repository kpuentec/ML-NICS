import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data():
    column_path = "data/KDDFeatureNames.txt"
    train_path = "data/KDDTrain+.txt"
    test_path = "data/KDDTest+.txt"

    with open(column_path, "r") as f:
        col_names = [line.strip().split(":")[0] for line in f.readlines()]
    col_names += ['class', 'difficulty']

    df_train = pd.read_csv(train_path, names=col_names)
    df_test = pd.read_csv(test_path, names=col_names)
    return df_train, df_test

def preprocess(df):
    df = df.copy()
    df.drop('difficulty', axis=1, inplace=True)

    for col in ['protocol_type', 'service', 'flag']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    df['class'] = df['class'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

    X = df.drop('class', axis=1)
    y = df['class']
    return X, y

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler