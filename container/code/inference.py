
import os
import json
import requests
import numpy as np
import pandas as pd

from pickle import load
from pathlib import Path


BASE_FOLDER = Path("/opt/ml/")
MODEL_DIRECTORY = BASE_FOLDER / "model"


def handler(data, context, model_directory=MODEL_DIRECTORY):
    """
    This is the entrypoint that will be called by SageMaker when the endpoint
    receives a request.
    
    You can see more information at https://github.com/aws/sagemaker-tensorflow-serving-container.
    """
    print("Handling endpoint request")
    
    instance = _process_input(data, context, model_directory)
    output = _predict(instance, context)
    return _process_output(output, context)


def transform(payload, model_directory):
    print("Transforming input data...")
    pipeline = load(open(model_directory / "pipeline.pkl", 'rb'))
    
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
    

def _process_input(data, context, model_directory):
    print("Processing input data...")
    
    if context is None:
        # The context will be None when we are testing the code
        # directly from a notebook. In that case, we can use the
        # data directly.
        endpoint_input = data
    elif context.request_content_type == "application/json":
        # When the endpoint is running, we will receive a context
        # object. We need to parse the input and turn it into 
        # JSON in that case.
        endpoint_input = json.loads(data.read().decode("utf-8"))

        if endpoint_input is None:
            raise ValueError("There was an error parsing the input request.")
    else:
        raise ValueError(f"Unsupported content type: {context.request_content_type or 'unknown'}")
        
    return transform(endpoint_input, model_directory)


def _predict(instance, context):
    print("Sending input data to model to make a prediction...")
    
    model_input = json.dumps({"instances": [instance]})
    
    if context is None:
        # The context will be None when we are testing the code
        # directly from a notebook. In that case, we want to return
        # a fake prediction back.
        result = {
            "predictions": [
                [0.2, 0.5, 0.3]
            ]
        }
    else:
        # When the endpoint is running, we will receive a context
        # object. In that case we need to send the instance to the
        # model to get a prediction back.
        response = requests.post(context.rest_uri, data=model_input)
        
        if response.status_code != 200:
            raise ValueError(response.content.decode('utf-8'))
            
        result = json.loads(response.content)
    
    print(f"Response {result}")
    return result


def _process_output(output, context):
    print("Processing prediction received from the model...")
    
    response_content_type = "application/json" if context is None else context.accept_header
    
    prediction = np.argmax(output["predictions"][0])
    confidence = output["predictions"][0][prediction]
    
    print(f"Prediction: {prediction}. Confidence: {confidence}")
    
    result = json.dumps({
        "prediction": str(prediction),
        "confidence": str(confidence)
    }), response_content_type
    
    return output
