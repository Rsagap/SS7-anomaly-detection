# model_server.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and scaler
try:
    model = joblib.load('final_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ Model and scaler loaded successfully")

    # Get feature names from training (you need to define them!)
    feature_columns = [
        'Unnamed: 0',
        'f_c_ossn_others',
        'f_same_cggt_is_gmlc_oc',
        'f_same_cggt_is_gmlc_ossn',
        'f_same_cggt_is_hlr_oc',
        'f_same_cggt_is_hlr_ossn',
        'f_velocity_greater_than_1000',
        'f_count_unloop_country_last_x_hours_ul',
        'f_count_gap_ok_sai_and_all_lu',
        'f_one_cggt_multi_cdgt_psi',
        'f_count_ok_cl_between2lu',
        'f_count_ok_dsd_between2lu',
        'f_count_ok_fwsm_mo_between2lu',
        'f_count_ok_fwsm_mt_between2lu',
        'f_count_ok_fwsm_report_between2lu',
        'f_count_ok_fwsm_submit_between2lu',
        'f_count_ok_isd_between2lu',
        'f_count_ok_prn_between2lu',
        'f_count_ok_psi_between2lu',
        'f_count_ok_purge_ms_between2lu',
        'f_count_ok_sai_between2lu',
        'f_count_ok_si_between2lu',
        'f_count_ok_sri_between2lu',
        'f_count_ok_srism_between2lu',
        'f_count_ok_ul_between2lu',
        'f_count_ok_ulgprs_between2lu',
        'f_count_ok_ussd_between2lu',
        'f_frequent_ok_cl_between2lu',
        'f_frequent_ok_dsd_between2lu',
        'f_frequent_ok_fwsm_mo_between2lu',
        'f_frequent_ok_fwsm_mt_between2lu',
        'f_frequent_ok_fwsm_report_between2lu',
        'f_frequent_ok_fwsm_submit_between2lu',
        'f_frequent_ok_isd_between2lu',
        'f_frequent_ok_prn_between2lu',
        'f_frequent_ok_psi_between2lu',
        'f_frequent_ok_purge_ms_between2lu',
        'f_frequent_ok_sai_between2lu',
        'f_frequent_ok_si_between2lu',
        'f_frequent_ok_sri_between2lu',
        'f_frequent_ok_srism_between2lu',
        'f_frequent_ok_ul_between2lu',
        'f_frequent_ok_ulgprs_between2lu',
        'f_frequent_ok_ussd_between2lu'
    ]
    print(f"📌 Expected {len(feature_columns)} features")

except Exception as e:
    print(f"❌ Error loading models: {e}")
    model = None
    scaler = None
    feature_columns = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None or feature_columns is None:
        return jsonify({'error': 'Model or scaler not loaded', 'status': 'failed'}), 500

    try:
        data = request.json

        # Build feature vector in correct order
        row = [data.get(col, 0.0) for col in feature_columns]  # default 0.0
        features_df = pd.DataFrame([row], columns=feature_columns)

        # Scale
        scaled_features = scaler.transform(features_df)

        # Predict
        prediction = int(model.predict(scaled_features)[0])
        confidence = float(model.predict_proba(scaled_features)[0][1])  # P(positive class)

        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
