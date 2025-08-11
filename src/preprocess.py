import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(labeled_path, unlabeled_path):
    # Load data
    labeled_df = pd.read_csv(labeled_path)
    unlabeled_df = pd.read_csv(unlabeled_path)

    # Drop identifier columns
    drop_cols = ['_time', 'c_timestamp', 'c_cggt', 'c_imsi']
    labeled_df.drop(columns=drop_cols, inplace=True)
    unlabeled_df.drop(columns=drop_cols, inplace=True)

    # Split features and labels
    X_labeled = labeled_df.drop(columns=['label'])
    y_labeled = labeled_df['label']

    # Drop label column from unlabeled data (if present)
    X_unlabeled = unlabeled_df.drop(columns=['label'], errors='ignore')

    # Normalize
    scaler = StandardScaler()
    X_labeled_scaled = scaler.fit_transform(X_labeled)
    X_unlabeled_scaled = scaler.transform(X_unlabeled)

    return X_labeled_scaled, y_labeled.values, X_unlabeled_scaled, scaler
