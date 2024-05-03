import json
import os

import joblib
import mlflow
import numpy as np


class Model(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        os.environ["KERAS_BACKEND"] = "jax"
        import keras

        print("Loading context...")

        self.features_pipeline = joblib.load(context.artifacts["features_transformer"])
        self.target_transformer = joblib.load(context.artifacts["target_transformer"])
        self.model = keras.saving.load_model(context.artifacts["model"])

    def predict(self, context, model_input, params=None):
        print("Handling request...")

        transformed_payload = self._process_input(model_input)
        output = self._predict(transformed_payload) if transformed_payload else None
        return self._process_output(output)

    def _process_input(self, payload):
        print("Transforming payload...")

        # Let's now transform the payload using the features pipeline.
        try:
            result = self.features_pipeline.transform(payload)
        except Exception as e:
            print(f"There was an error processing the payload. {e}")
            return None

        return result[0].tolist()

    def _predict(self, instance):
        print("Making a prediction using the transformed payload...")

        predictions = self.model.predict(np.array([instance]))
        result = {"predictions": predictions.tolist()}
        print(result)

        return result

    def _process_output(self, output):
        print("Processing prediction received from the model...")

        if output:
            prediction = np.argmax(output["predictions"][0])
            confidence = output["predictions"][0][prediction]

            classes = self.target_transformer.named_transformers_[
                "species"
            ].categories_[0]

            result = {
                "prediction": classes[prediction],
                "confidence": confidence,
            }
        else:
            result = {"prediction": None}

        print(result)

        return json.dumps(result)
