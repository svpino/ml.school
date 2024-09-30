import logging

from common import PYTHON, FlowMixin, configure_logging, packages
from metaflow import (
    FlowSpec,
    Parameter,
    project,
    pypi_base,
    step,
)

configure_logging()


@project(name="penguins")
@pypi_base(
    python=PYTHON,
    packages=packages("pandas", "numpy", "boto3", "requests"),
)
class Traffic(FlowSpec, FlowMixin):
    """A traffic generation pipeline that sends traffic to a running model.

    This pipeline will send fake traffic to a hosted model. It uses the original dataset
    to send random samples to the model.
    """

    target = Parameter(
        "target",
        help=(
            "The target platform hosting the model and where the traffic will be sent. "
            "The supported values are 'local' for models hosted as a local inference "
            "service and 'sagemaker' for models hosted on a SageMaker endpoint."
        ),
        default="local",
    )

    endpoint_uri = Parameter(
        "endpoint-uri",
        help=(
            "The URI of the hosted model. If the target is a model hosted in "
            "SageMaker, this parameter will point the URI of the SageMaker endpoint."
        ),
        required=True,
    )

    samples = Parameter(
        "samples",
        help="The number of samples that will be sent to the hosted model.",
        default=200,
    )

    drift = Parameter(
        "drift",
        help=(
            "Whether to introduce drift in the samples submitted to the model. This is "
            "useful for testing the monitoring process."
        ),
        default=False,
    )

    @step
    def start(self):
        """Start the pipeline and load the dataset."""
        self.data = self.load_dataset()

        if self.target not in ["local", "sagemaker"]:
            message = "The specified target is not supported."
            raise RuntimeError(message)

        self.next(self.prepare_data)

    @step
    def prepare_data(self):
        """Prepare the data and introduce drift before submitting it to the model."""
        import numpy as np

        self.data.pop("species")
        self.data["sex"] = self.data["sex"].replace(".", np.nan)
        self.data = self.data.dropna()

        # If we want to introduce drift, we will add random noise to one of the
        # numerical features in the data.
        if self.drift:
            rng = np.random.default_rng()
            self.data["body_mass_g"] += rng.uniform(
                1,
                3 * self.data["body_mass_g"].std(),
                size=len(self.data),
            )

        self.next(self.traffic)

    @step
    def traffic(self):
        """Prepare the payload and send traffic to the hosted model."""
        import boto3
        import pandas as pd

        if self.target == "sagemaker":
            sagemaker_runtime = boto3.Session().client("sagemaker-runtime")

        self.dispatched_samples = 0
        self.predictions = []

        try:
            while self.dispatched_samples < self.samples:
                payload = {}

                batch = self.data.sample(n=10)
                payload["inputs"] = [
                    {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
                    for _, row in batch.iterrows()
                ]

                if self.target == "local":
                    predictions = self._invoke_local_endpoint(payload)
                elif self.target == "sagemaker":
                    predictions = self._invoke_sagemaker_endpoint(
                        sagemaker_runtime,
                        payload,
                    )

                self.predictions.append(predictions)
                self.dispatched_samples += len(batch)
        except Exception:
            logging.exception("There was an error sending traffic to the endpoint.")

        self.next(self.end)

    @step
    def end(self):
        """End of the pipeline."""
        for batch in self.predictions:
            for prediction in batch["predictions"]:
                logging.info(
                    "Sample: [Prediction: %s. Confidence: %.2f]",
                    prediction["prediction"],
                    prediction["confidence"],
                )

        logging.info(
            "Dispatched %s samples to the hosted model.",
            self.dispatched_samples,
        )

    def _invoke_local_endpoint(self, payload):
        """Submit the given payload to a local inference service."""
        import json

        import requests

        payload["params"] = {"data_capture": True}
        predictions = requests.post(
            url=self.endpoint_uri,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=5,
        )

        return predictions.json()

    def _invoke_sagemaker_endpoint(self, sagemaker_runtime, payload):
        """Submit the given payload to a SageMaker endpoint."""
        import json

        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=self.endpoint_uri,
            ContentType="application/json",
            Body=json.dumps(payload),
        )

        return json.loads(response["Body"].read().decode())


if __name__ == "__main__":
    Traffic()
