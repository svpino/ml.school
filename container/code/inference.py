import os
import json
import requests
import numpy as np
import pandas as pd

from pickle import load
from pathlib import Path


BASE_FOLDER = Path("/opt/ml/")
MODEL_DIRECTORY = BASE_FOLDER / "model"


def handler(data, context):
    print("Handler called")
    
    instance = _process_input(data, context)
    response = _predict(instance, context)
    return _process_output(response, context)


def transform(payload):
    pipeline = load(open(MODEL_DIRECTORY / "pipeline.pkl", 'rb'))
    
    island = payload.get("island", "Biscoe")
    culmen_length_mm = payload.get("culmen_length_mm", 0)
    culmen_depth_mm = payload.get("culmen_depth_mm", 0)
    flipper_length_mm = payload.get("flipper_length_mm", 0)
    body_mass_g = payload.get("body_mass_g", 0)
    
    data = pd.DataFrame(
        columns=["island", "culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"], 
        data=[[
            island, 
            culmen_length_mm, 
            culmen_depth_mm, 
            flipper_length_mm, 
            body_mass_g
        ]]
    )
    
    result = pipeline.transform(data)
    return result[0].tolist()
    

def _process_input(data, context):
    print("Called _process_input...")
    
    if context.request_content_type == "application/json":
        data = json.loads(data.read().decode("utf-8"))

        if data is None:
            raise ValueError("There was an error parsing the input request.")
            
        return transform(data)

    raise ValueError('{{"error": "unsupported content type {}"}}'.format(
        context.request_content_type or "unknown"))


def _predict(instance, context):
    print("Called _predict...")
    
    model_input = json.dumps({"instances": [instance]})
    
    response = requests.post(context.rest_uri, data=model_input)
    
    print(f"Response {response}")
    return response


def _process_output(response, context):
    print("Called _process_output...")
    
    if response.status_code != 200:
        raise ValueError(response.content.decode('utf-8'))

    response_content_type = context.accept_header
    result = json.loads(response.content)
    print(result)
    
    prediction = np.argmax(result["predictions"][0])
    confidence = result["predictions"][0][prediction]
    
    print(f"Prediction: {prediction}, Confidence: {confidence}")
    
    result = json.dumps({
        "prediction": str(prediction),
        "confidence": str(confidence)
    }), response_content_type
    
    return result

