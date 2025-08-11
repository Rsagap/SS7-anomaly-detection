import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

def train_final_model(X_labeled, y_labeled, X_pseudo, y_pseudo, save_path):
    # Combine labeled and high-confidence pseudo-labeled
    X_all = np.vstack((X_labeled, X_pseudo))
    y_all = np.concatenate((y_labeled, y_pseudo))

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_all, y_all)

    # Save model
    dump(clf, save_path)
    return clf
