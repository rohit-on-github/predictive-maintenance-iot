import joblib
import json
import numpy as np

def init():
    global model
    # Load the model from the file
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'rf_model.pkl')
    model = joblib.load(model_path)

def run(data):
    try:
        input_data = np.array(json.loads(data)['data'])
        result = model.predict(input_data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
