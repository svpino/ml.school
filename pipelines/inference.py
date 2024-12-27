import importlib
import logging
import logging.config
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.pyfunc.model import PythonModelContext


class Model(mlflow.pyfunc.PythonModel):
    """A custom model implementing an inference pipeline to classify penguins.

    This inference pipeline has three phases: preprocessing, prediction, and
    postprocessing. The model will optionally store the input requests and predictions
    in a SQLite database.

    The [Custom MLflow Models with mlflow.pyfunc](https://mlflow.org/blog/custom-pyfunc)
    blog post is a great reference to understand how to use custom Python models in
    MLflow.
    """

    def __init__(self):
        self.backend = None

    def load_context(self, context: PythonModelContext) -> None:
        """Load and prepare the model context to make predictions.

        This function is called only once as soon as the model is constructed. It loads
        the transformers and the Keras model specified as artifacts.
        """
        # By default, we want to use the JAX backend for Keras.
        if not os.getenv("KERAS_BACKEND"):
            os.environ["KERAS_BACKEND"] = "jax"

        import keras

        logging.info("Keras backend: %s", os.environ.get("KERAS_BACKEND"))

        self._configure_logging()
        logging.info("Loading model context...")

        # endpoint_config_file = os.getenv("ENDPOINT_CONFIG", "sqlite.json")
        # try:
        #     with Path(context.artifacts[endpoint_config_file]).open() as f:
        #         endpoint_config = json.loads(
        #             f.read(),
        #             object_hook=lambda d: SimpleNamespace(**d),
        #         )

        #     module, cls = endpoint_config.endpoint.rsplit(".", 1)
        #     module = importlib.import_module(module)
        #     self.endpoint = getattr(module, cls)(endpoint_config)
        # except Exception:
        #     logging.exception(
        #         "There was an error instantiating the endpoint class.",
        #     )

        backend_class = os.getenv("MODEL_BACKEND", "backend.SQLite")

        try:
            module, cls = backend_class.rsplit(".", 1)
            module = importlib.import_module(module)
            self.backend = getattr(module, cls)()
        except Exception:
            logging.exception(
                "There was an error instantiating the endpoint class.",
            )

        logging.info(
            "Backend: %s",
            type(self.backend).__name__ if self.backend else None,
        )

        # First, we need to load the transformation pipelines from the model artifacts.
        # These will help us transform the input data and the output predictions.
        self.features_transformer = joblib.load(
            context.artifacts["features_transformer"],
        )
        self.target_transformer = joblib.load(context.artifacts["target_transformer"])

        # Then, we can load the Keras model we trained.
        self.model = keras.saving.load_model(context.artifacts["model"])

        logging.info("Model is ready to receive requests")

    def predict(
        self,
        context: PythonModelContext,  # noqa: ARG002
        model_input: pd.DataFrame | list[dict[str, Any]] | dict[str, Any] | list[Any],
        params: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> list:
        """Handle the request received from the client.

        This method is responsible for processing the input data received from the
        client, making a prediction using the model, and returning a readable response
        to the client.
        """
        if isinstance(model_input, list):
            model_input = pd.DataFrame(model_input)

        if isinstance(model_input, dict):
            model_input = pd.DataFrame([model_input])

        logging.info(
            "Received prediction request with %d %s",
            len(model_input),
            "samples" if len(model_input) > 1 else "sample",
        )

        model_output = []

        transformed_payload = self.process_input(model_input)
        if transformed_payload is not None:
            logging.info("Making a prediction using the transformed payload...")
            predictions = self.model.predict(transformed_payload, verbose=0)

            model_output = self.process_output(predictions)

        if self.backend is not None:
            self.backend.save(model_input, model_output)

        logging.info("Returning prediction to the client")
        logging.debug("%s", model_output)

        return model_output

    def process_input(self, payload: pd.DataFrame) -> pd.DataFrame | None:
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
        except Exception:
            logging.exception("There was an error processing the payload.")
            return None

        return result

    def process_output(self, output: np.ndarray) -> list[dict[str, Any]]:
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
            # Notice that we need to unwrap the numpy values so we can serialize the
            # output as JSON.
            result = [
                {"prediction": p.item(), "confidence": c.item()}
                for p, c in zip(prediction, confidence, strict=True)
            ]

        return result

    def _configure_logging(self):
        """Configure how the logging system will behave."""
        import sys

        if Path("logging.conf").exists():
            logging.config.fileConfig("logging.conf")
        else:
            logging.basicConfig(
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[logging.StreamHandler(sys.stdout)],
                level=logging.INFO,
            )
