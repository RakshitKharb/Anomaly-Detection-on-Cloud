import json
import joblib
import numpy as np
import os

def init():
    global model
    # Load the model from the same directory where the scoring script is located
    model_path = os.path.join(os.getcwd(), "model.pkl")
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        input_features = np.array(data["input"]).reshape(1, -1)
        prediction = model.predict(input_features)
        return json.dumps({"prediction": prediction.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
