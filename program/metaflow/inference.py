# Important documentation: https://mlflow.org/blog/custom-pyfunc

import os

import joblib
import mlflow
import numpy as np


class Model(mlflow.pyfunc.PythonModel):
    def __init__(self, configuration: dict):
        self.configuration = configuration

    def load_context(self, context):
        os.environ["KERAS_BACKEND"] = "jax"
        import keras

        print("Loading model context...")

        # First, we need to load the transformers from the artifacts.
        self.features_pipeline = joblib.load(context.artifacts["features_transformer"])
        self.target_transformer = joblib.load(context.artifacts["target_transformer"])

        # Then, we can load the model.
        self.model = keras.saving.load_model(context.artifacts["model"])

        print("Model is ready to receive requests...")

    def predict(self, context, model_input, params=None):
        print("Handling request...")

        print("Here is the config:", self.configuration)

        transformed_payload = self._process_input(model_input)
        output = (
            self._predict(transformed_payload)
            if transformed_payload is not None
            else None
        )
        return self._process_output(output)

    def _process_input(self, payload):
        print("Transforming payload...")

        # We need to transform the payload using the transformer. This can raise an
        # exception if the payload is not valid, in which case we should return None
        # to indicate that the prediction should not be made.
        try:
            result = self.features_pipeline.transform(payload)
        except Exception as e:
            print(f"There was an error processing the payload. {e}")
            return None

        return result

    def _predict(self, payload):
        print("Making a prediction using the transformed payload...")

        return self.model.predict(payload)

    def _process_output(self, output):
        print("Processing prediction received from the model...")

        result = {}
        if output is not None:
            prediction = np.argmax(output, axis=1)
            confidence = np.max(output, axis=1)

            # Let's transform the prediction index back to the
            # original species. We can use the target transformer
            # to access the list of classes.
            classes = self.target_transformer.named_transformers_[
                "species"
            ].categories_[0]
            prediction = np.vectorize(lambda x: classes[x])(prediction)

            # We can now return the prediction and the confidence
            # from the model.
            result = [
                {"prediction": p, "confidence": c}
                for p, c in zip(prediction, confidence, strict=True)
            ]

        return result
