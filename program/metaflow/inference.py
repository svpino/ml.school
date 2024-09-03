# Important documentation: https://mlflow.org/blog/custom-pyfunc

import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Model(mlflow.pyfunc.PythonModel):
    """A custom model that can be used to make predictions.

    This model implements an inference pipeline with three phases: preprocessing,
    prediction, and postprocessing. The model will optionally store the input requests
    and predictions in a SQLite database.
    """

    def __init__(
        self,
        data_capture_file: str | None = "penguins.db",
        *,
        data_capture: bool = False,
    ) -> None:
        """Initialize the model.

        By default, the model will not store the input requests and predictions. This
        behavior can be overwritten on every individual request.

        This constructor expects the filename that will be used to create a SQLite
        database to store the input requests and predictions. If no filename is
        specified, the model will use "penguins.db" as the default name. You can
        override the default database filename by setting the `MODEL_DATA_CAPTURE_FILE`
        environment variable.
        """
        self.data_capture = data_capture
        self.data_capture_file = data_capture_file

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the transformers and the Keras model specified as artifacts.

        This function is called only once as soon as the model is constructed.
        """
        os.environ["KERAS_BACKEND"] = "jax"
        import keras

        logging.info("Loading model context...")

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

        logging.info("Model is ready to receive requests...")

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,  # noqa: ARG002
        model_input,
        params: dict[str, Any] | None = None,
    ) -> list:
        """Handle the request received from the client.

        This method is responsible for processing the input data received from the
        client, making a prediction using the model, and returning a readable response
        to the client.

        The caller can specify whether we should capture the input request and
        prediction by using the `data_capture` parameter when making a request.
        """
        logging.info("Handling request...")

        if isinstance(model_input, list | dict):
            model_input = pd.DataFrame(model_input)

        model_output = []

        transformed_payload = self.process_input(model_input)
        if transformed_payload is not None:
            logging.info("Making a prediction using the transformed payload...")
            predictions = self.model.predict(transformed_payload)

            model_output = self.process_output(predictions)

        # If the caller specified the `data_capture` parameter when making the
        # request, we should use it to determine whether we should capture the
        # input request and prediction.
        if (
            params
            and params.get("data_capture", False) is True
            or not params
            and self.data_capture
        ):
            self.capture(model_input, model_output)

        return model_output

    def process_input(self, payload: pd.DataFrame) -> pd.DataFrame:
        """Process the input data received from the client.

        This method is responsible for transforming the input data received from the
        client into a format that can be used by the model.
        """
        logging.info("Transforming payload...")

        # We need to transform the payload using the transformer. This can raise an
        # exception if the payload is not valid, in which case we should return None
        # to indicate that the prediction should not be made.
        try:
            result = self.features_transformer.transform(payload)
        except Exception as e:
            logging.info("There was an error processing the payload. %s", e)
            return None

        return result

    def process_output(self, output: np.ndarray) -> list:
        """Process the prediction received from the model.

        This method is responsible for transforming the prediction received from the
        model into a readable format that will be returned to the client.
        """
        logging.info("Processing prediction received from the model...")

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

            # We can now return the prediction and the confidence from the model.
            # Notice that we need to unwrap the confidence's numpy value to return
            # a float so that it can be serialized to JSON.
            result = [
                {"prediction": p, "confidence": c.item()}
                for p, c in zip(prediction, confidence, strict=True)
            ]

        return result

    def capture(self, model_input: pd.DataFrame, model_output: list) -> None:
        """Save the input request and output prediction to the database.

        This method will save the input request and output prediction to a SQLite
        database. If the database doesn't exist, this function will create it.
        """
        logging.info("Saving input request and output prediction to the database...")

        # If the MODEL_DATA_CAPTURE_FILE environment variable is set, we should use it
        # to specify the database filename. Otherwise, we'll use the default filename
        # specified when the model was instantiated.
        data_capture_file = os.environ.get(
            "MODEL_DATA_CAPTURE_FILE",
            self.data_capture_file,
        )

        connection = None
        try:
            connection = sqlite3.connect(data_capture_file)

            data = model_input.copy()

            # We need to add the current time, the prediction and confidence columns
            # to the DataFrame to store everything together.
            data["date"] = datetime.now(timezone.utc)

            # Let's initialize the prediction and confidence columns with None. We'll
            # overwrite them later if the model output is not empty.
            data["prediction"] = None
            data["confidence"] = None

            # Let's also add a column to store the ground truth. This column can be
            # used by the labeling team to provide the actual species for the data.
            data["species"] = None

            # If the model output is not empty, we should update the prediction and
            # confidence columns with the corresponding values.
            if model_output is not None and len(model_output) > 0:
                data["prediction"] = [item["prediction"] for item in model_output]
                data["confidence"] = [item["confidence"] for item in model_output]

            # Finally, we can save the data to the database.
            data.to_sql("data", connection, if_exists="append", index=False)

        except sqlite3.Error as e:
            logging.info(
                "There was an error saving the input request and output prediction "
                "in the database. %s",
                e,
            )
        finally:
            if connection:
                connection.close()
