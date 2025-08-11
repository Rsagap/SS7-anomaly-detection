import requests
import random
import time
from datetime import datetime, timedelta

# Configuration
API_URL = "http://localhost:5000/predict"
NORMAL_REQUESTS = 100
ATTACK_REQUESTS = 20
DELAY_BETWEEN_REQUESTS = 0.1  # seconds

def generate_timestamp():
    return (datetime.now() - timedelta(seconds=random.randint(0, 86400))).isoformat() + "+01:00"

def generate_normal_traffic():
    return {
        'Unnamed: 0': 0,
        '_time': generate_timestamp(),
        'c_cggt': random.choice([11111111, 22222222]),
        'c_imsi': int(2.42011e14) + random.randint(0, 1000),
        'f_c_ossn_others': 0,
        'f_same_cggt_is_gmlc_oc': 0,
        'f_same_cggt_is_gmlc_ossn': 0,
        'f_same_cggt_is_hlr_oc': random.choice([0, 1]),
        'f_same_cggt_is_hlr_ossn': random.choice([0, 1]),
        'f_velocity_greater_than_1000': random.choices([0, 1], weights=[0.9, 0.1])[0],
        # Include all features with normal values
        # ... (add all remaining features with normal ranges)
    }

def generate_attack_traffic():
    data = generate_normal_traffic()
    # Modify specific features to create attack patterns
    data.update({
        'f_velocity_greater_than_1000': 1,
        'f_count_unloop_country_last_x_hours_ul': random.randint(3, 5),
        'f_count_ok_cl_between2lu': random.randint(5, 10),
        'f_count_ok_sri_between2lu': random.randint(5, 8),
        # Modify other features that indicate attacks
    })
    return data

def send_request(data):
    try:
        response = requests.post(API_URL, json=data, timeout=5)
        result = response.json()
        
        if result.get('status') == 'success':
            print(f"IMSI: {data['c_imsi']} | Prediction: {result['prediction']} | Confidence: {result['confidence']:.2%}")
            if result['prediction'] == 1:
                print("🚨 FRAUD DETECTED!")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Request failed: {e}")

def run_test():
    print("Starting normal traffic test...")
    for _ in range(NORMAL_REQUESTS):
        data = generate_normal_traffic()
        send_request(data)
        time.sleep(DELAY_BETWEEN_REQUESTS)
    
    print("\nStarting attack traffic test...")
    for _ in range(ATTACK_REQUESTS):
        data = generate_attack_traffic()
        send_request(data)
        time.sleep(DELAY_BETWEEN_REQUESTS)

if __name__ == "__main__":
    run_test()
