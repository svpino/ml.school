import logging

from common import PYTHON, DatasetMixin, configure_logging, load_instance, packages
from metaflow import (
    FlowSpec,
    Parameter,
    conda_base,
    project,
    step,
)

configure_logging()


@project(name="penguins")
@conda_base(
    python=PYTHON,
    packages=packages("pandas", "numpy", "boto3", "requests"),
)
class Traffic(FlowSpec, DatasetMixin):
    """A pipeline for sending fake traffic to a hosted model."""

    endpoint = Parameter(
        "endpoint",
        help=(
            "Name of the class implementing the `endpoint.Endpoint` abstract class. "
            "This class is responsible for sending a payload to a hosted model."
        ),
        default="endpoint.Local",
    )

    target = Parameter(
        "target",
        help=(
            "The location of the hosted model where the pipeline will send the traffic."
        ),
        default="http://127.0.0.1:8080/invocations",
    )

    samples = Parameter(
        "samples",
        help="The number of samples that will be sent to the hosted model.",
        default=200,
        required=False,
    )

    drift = Parameter(
        "drift",
        help=(
            "Whether to introduce drift in the samples sent to the model. This is "
            "useful for testing the monitoring process."
        ),
        default=False,
        required=False,
    )

    @step
    def start(self):
        """Start the pipeline and load the dataset."""
        self.data = self.load_dataset()

        # Let's instantiate the endpoint class that will be responsible for sending
        # the traffic to the hosted model.
        try:
            self.endpoint_impl = load_instance(
                name=self.endpoint,
                target=self.target,
            )

            logging.info("Endpoint: %s", type(self.endpoint).__name__)
        except Exception as e:
            message = "There was an error instantiating the endpoint class."
            logging.exception(message)
            raise RuntimeError(message) from e

        self.next(self.prepare_data)

    @step
    def prepare_data(self):
        """Prepare the data and introduce drift before submitting it to the model."""
        import numpy as np

        # We don't need to send the target column to the model.
        self.data.pop("species")
        self.data = self.data.dropna()

        # If we want to introduce drift, we can add random noise to one of the
        # numerical features in the data.
        if self.drift:
            rng = np.random.default_rng()
            self.data["body_mass_g"] += rng.uniform(
                1,
                3 * self.data["body_mass_g"].std(),
                size=len(self.data),
            )

        self.next(self.generate_traffic)

    @step
    def generate_traffic(self):
        """Prepare the payload and send traffic to the hosted model."""
        import pandas as pd

        self.dispatched_samples = 0

        try:
            while self.dispatched_samples < self.samples:
                batch = min(self.samples - self.dispatched_samples, 10)
                payload = {}

                batch = self.data.sample(n=batch)
                payload["inputs"] = [
                    {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
                    for _, row in batch.iterrows()
                ]

                self.endpoint_impl.invoke(payload)
                self.dispatched_samples += len(batch)
        except Exception:
            logging.exception("There was an error sending traffic to the endpoint.")

        self.next(self.end)

    @step
    def end(self):
        """End of the pipeline."""
        logging.info(
            "Dispatched %s samples to the hosted model.",
            self.dispatched_samples,
        )


if __name__ == "__main__":
    Traffic()
