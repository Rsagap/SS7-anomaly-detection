import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score
from src.preprocess import load_and_preprocess

# Paths
TEST_PATH = "data/Test_data.csv"  # <-- create or specify your labeled test CSV
MODEL_PATH = "models/final_model.pkl"
SCALER_PATH = "models/scaler.pkl"  # we'll save scaler during training

def main():
    # Load trained model & scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Load test data
    test_df = pd.read_csv(TEST_PATH)

    # Match preprocessing: drop identifier columns
    drop_cols = ['_time', 'c_timestamp', 'c_cggt', 'c_imsi']
    test_df.drop(columns=drop_cols, inplace=True)

    # Separate features & labels
    X_test = test_df.drop(columns=['label'])
    y_test = test_df['label']

    # Scale with training scaler
    X_test_scaled = scaler.transform(X_test)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Report
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
