import requests
import json
import pandas as pd

SERVING_URL = "http://127.0.0.1:5001/invocations"

test_data = pd.read_csv("heart_preprocessing/heart_test_preprocessed.csv")
X_test = test_data.drop('target', axis=1)
y_test = test_data['target']

sample = X_test.iloc[0:5]

payload = {
    "dataframe_split": {
        "columns": sample.columns.tolist(),
        "data": sample.values.tolist()
    }
}

headers = {"Content-Type": "application/json"}

response = requests.post(SERVING_URL, headers=headers, data=json.dumps(payload))

if response.status_code == 200:
    predictions = response.json()
    print(f"Predictions: {predictions['predictions']}")
    print(f"Actual: {y_test.iloc[0:5].tolist()}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
