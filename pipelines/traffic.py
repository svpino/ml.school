import logging
import sys

from common import PYTHON, FlowMixin, packages
from metaflow import (
    FlowSpec,
    Parameter,
    project,
    pypi_base,
    step,
)

logger = logging.getLogger(__name__)


@project(name="penguins")
@pypi_base(
    python=PYTHON,
    packages=packages("pandas", "numpy", "boto3", "requests"),
)
class Traffic(FlowSpec, FlowMixin):
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
        help="The URI of the endpoint to be used.",
        required=True,
    )

    drift = Parameter(
        "drift",
        help="Whether to introduce drift in the data.",
        default=False,
    )

    @step
    def start(self):
        self.data = self.load_dataset()

        if self.target not in ["local", "sagemaker"]:
            message = "The specified target is not supported."
            raise RuntimeError(message)

        self.next(self.prepare_data)

    @step
    def prepare_data(self):
        import numpy as np

        self.data.pop("species")
        self.data["sex"] = self.data["sex"].replace(".", np.nan)

        if self.drift:
            std_dev = self.data["body_mass_g"].std()
            rng = np.random.default_rng()
            self.data["body_mass_g"] += rng.uniform(1, 3 * std_dev, size=len(self.data))

        self.data = self.data.sample(frac=1).reset_index(drop=True)

        self.next(self.traffic)

    @step
    def traffic(self):
        import boto3
        import pandas as pd

        def nan_to_none(value):
            return None if pd.isna(value) else value

        if self.target == "sagemaker":
            sagemaker_runtime = boto3.Session().client("sagemaker-runtime")

        self.predictions = []

        try:
            payload = {}
            for batch_index in range(0, len(self.data), 10):
                batch = self.data[batch_index : batch_index + 10]

                samples = [
                    {k: nan_to_none(v) for k, v in row.to_dict().items()}
                    for _, row in batch.iterrows()
                ]

                payload["inputs"] = samples

                if self.target == "local":
                    predictions = self._invoke_local_endpoint(payload)
                elif self.target == "sagemaker":
                    predictions = self._invoke_sagemaker_endpoint(
                        sagemaker_runtime,
                        payload,
                    )

                self.predictions.append(predictions)
        except Exception:
            logger.exception("There was an error sending traffic to the endpoint.")

        self.next(self.end)

    @step
    def end(self):
        for batch in self.predictions:
            for prediction in batch["predictions"]:
                logger.info(
                    "Prediction: %s. Confidence: %.2f",
                    prediction["prediction"],
                    prediction["confidence"],
                )

    def _invoke_local_endpoint(self, payload):
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
        import json

        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=self.endpoint_uri,
            ContentType="application/json",
            Body=json.dumps(payload),
        )

        return json.loads(response["Body"].read().decode())


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    Traffic()
