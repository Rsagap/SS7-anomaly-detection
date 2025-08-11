import numpy as np
from sklearn.semi_supervised import LabelSpreading

def train_label_spreading(X_labeled, y_labeled, X_unlabeled):
    # Combine
    X_combined = np.vstack((X_labeled, X_unlabeled))
    y_combined = np.concatenate((y_labeled, [-1] * len(X_unlabeled)))

    # Label Spreading
    model = LabelSpreading(kernel='rbf', alpha=0.2)
    model.fit(X_combined, y_combined)

    # Predict
    full_labels = model.transduction_
    pseudo_labels = full_labels[len(X_labeled):]
    return pseudo_labels, model
