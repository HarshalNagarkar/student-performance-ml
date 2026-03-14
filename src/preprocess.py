import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_raw_data(filepath="data/student_data.csv"):
    """Load raw CSV data."""
    df = pd.read_csv(filepath)
    return df

def handle_missing_values(df):
    """Drop rows with missing values."""
    df = df.dropna()
    return df

def remove_duplicates(df):
    """Remove duplicate rows."""
    df = df.drop_duplicates()
    return df

def validate_ranges(df):
    """Remove rows with values outside valid ranges."""
    df = df[df["attendance"].between(0, 100)]
    df = df[df["internal_marks"].between(0, 50)]
    df = df[df["assignment_score"].between(0, 50)]
    return df

def check_empty_data(df):
    """Raise error if dataframe is empty after preprocessing."""
    if df.empty:
        raise ValueError("Dataset is empty after preprocessing.")
    return df

def encode_labels(df):
    """Encode the result column to numeric."""
    encoder = LabelEncoder()
    df["result_encoded"] = encoder.fit_transform(df["result"])
    return df, encoder

def scale_features(X_train, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def preprocess_pipeline(filepath="data/student_data.csv", test_size=0.2):
    """Full preprocessing pipeline."""
    df = load_raw_data(filepath)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = validate_ranges(df)
    df = check_empty_data(df)
    df, encoder = encode_labels(df)

    X = df[["attendance", "internal_marks", "assignment_score"]]
    y = df["result_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print("Preprocessing complete.")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    return X_train_scaled, X_test_scaled, y_train, y_test, encoder, scaler

if __name__ == "__main__":
    preprocess_pipeline()
