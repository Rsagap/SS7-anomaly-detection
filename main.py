from src.preprocess import load_and_preprocess
from src.semi_supervised_train import train_label_spreading
from src.supervised_train import train_final_model
import numpy as np
import os
from joblib import dump

# File paths
LABELED_PATH = 'data/Labeled_data.csv'
UNLABELED_PATH = 'data/Unlabeled_data.csv'
MODEL_SAVE_PATH = 'models/final_model.pkl'

def main():
    # Step 1: Preprocess
    print("[*] Loading and preprocessing data...")
    X_labeled, y_labeled, X_unlabeled, scaler = load_and_preprocess(LABELED_PATH, UNLABELED_PATH)
    dump(scaler, "models/scaler.pkl")

    # Step 2: Semi-supervised label spreading
    print("[*] Running Label Spreading...")
    pseudo_labels, ss_model = train_label_spreading(X_labeled, y_labeled, X_unlabeled)

    # Optional: Filter high confidence predictions (prob > 0.9)
    probs = ss_model.label_distributions_[len(X_labeled):]
    mask = probs.max(axis=1) > 0.9
    X_confident = X_unlabeled[mask]
    y_confident = pseudo_labels[mask]

    print(f"[*] {len(y_confident)} confident pseudo-labeled samples found.")

    # Step 3: Supervised training
    print("[*] Training final RandomForest model...")
    os.makedirs("models", exist_ok=True)
    clf = train_final_model(X_labeled, y_labeled, X_confident, y_confident, MODEL_SAVE_PATH)

    print("[✓] Model trained and saved to", MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
