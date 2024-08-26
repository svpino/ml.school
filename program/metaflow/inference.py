# Important documentation: https://mlflow.org/blog/custom-pyfunc

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd


class Model(mlflow.pyfunc.PythonModel):
    """A custom model that can be used to make predictions.

    This model implements an inference pipeline with three phases: preprocessing,
    prediction, and postprocessing.
    """

    def __init__(self, database: str | None = None) -> None:
        """Initialize the model.

        This constructor expects the filename that will be used to create a SQLite
        database to store the input requests and predictions. If no filename is
        specified, the model will use "penguins.db" as the default name.

        You can override the default database filename by setting the
        `MODEL_DATABASE` environment variable.
        """
        self.database = database or "penguins.db"

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the transformers and the Keras model specified as artifacts.

        This function is called only once as soon as the model is constructed.
        """
        os.environ["KERAS_BACKEND"] = "jax"
        import keras

        print("Loading model context...")

        # First, we need to load the transformation pipelines from the artifacts. These
        # will help us transform the input data and the output predictions. Notice that
        # these transformation pipelines are the ones we fitted during the training
        # phase.
        self.features_transformer = joblib.load(
            context.artifacts["features_transformer"],
        )
        self.target_transformer = joblib.load(context.artifacts["target_transformer"])

        # Then, we can load the Keras model we trained.
        self.model = keras.saving.load_model(context.artifacts["model"])

        print("Model is ready to receive requests...")

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input,
        params: dict[str, Any] | None = None,
    ) -> list:
        """Handle the request received from the client.

        This method is responsible for processing the input data received from the
        client, making a prediction using the model, and returning a readable response
        to the client.
        """
        print("Handling request...")

        if isinstance(model_input, list | dict):
            model_input = pd.DataFrame(model_input)

        transformed_payload = self.process_input(model_input)
        if transformed_payload is None:
            return []

        print("Making a prediction using the transformed payload...")
        predictions = self.model.predict(transformed_payload)
        if predictions is None:
            return []

        model_output = self.process_output(predictions)

        # We will only store the input request and prediction if the capture parameter
        # is set to True.
        if params and params.get("capture", False) is True:
            self.capture(model_input, model_output)

        return model_output

    def process_input(self, payload: pd.DataFrame) -> pd.DataFrame:
        """Process the input data received from the client.

        This method is responsible for transforming the input data received from the
        client into a format that can be used by the model.
        """
        print("Transforming payload...")

        # We need to transform the payload using the transformer. This can raise an
        # exception if the payload is not valid, in which case we should return None
        # to indicate that the prediction should not be made.
        try:
            result = self.features_transformer.transform(payload)
        except Exception as e:
            print(f"There was an error processing the payload. {e}")
            return None

        return result

    def process_output(self, output: np.ndarray) -> list:
        """Process the prediction received from the model.

        This method is responsible for transforming the prediction received from the
        model into a readable format that will be returned to the client.
        """
        print("Processing prediction received from the model...")

        result = []
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

    def capture(self, model_input: pd.DataFrame, model_output: list) -> None:
        """Save the input request and output prediction to the database.

        This method will save the input request and output prediction to a SQLite
        database specified when the model was instantiated. If the database doesn't
        exist, this function will create it.
        """
        print("Saving input request and output prediction to the database...")

        # If the MODEL_DATABASE environment variable is set, we should use it to
        # specify the database filename. Otherwise, we'll use the default filename
        # specified when the model was instantiated.
        database_path = os.environ.get("MODEL_DATABASE", self.database)

        connection = None
        try:
            connection = sqlite3.connect(database_path)

            data = model_input.copy()

            # We need to add the current time, the prediction and confidence columns
            # to the DataFrame to store everything together.
            data["date"] = datetime.now(timezone.utc)
            data["prediction"] = [item["prediction"] for item in model_output]
            data["confidence"] = [item["confidence"] for item in model_output]

            data.to_sql("data", connection, if_exists="append", index=False)

        except sqlite3.Error as e:
            print(
                "There was an error saving the input request and output prediction "
                f"in the database. {e}",
            )
        finally:
            if connection:
                connection.close()
