from common import Pipeline, backend, dataset
from metaflow import (
    Parameter,
    step,
)


class Traffic(Pipeline):
    """A pipeline for sending fake traffic to a hosted model."""

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

    @dataset
    @backend
    @step
    def start(self):
        """Start the pipeline."""
        import numpy as np

        # We don't need to send the target column to the model,
        # and we don't want to use samples with missing values,
        # so let's get rid of them.
        self.data = self.data.drop(columns=["species"]).dropna()

        # If we want to introduce drift, we can add random noise to
        # one of the numerical features in the data. This is helpful
        # for testing the Monitoring pipeline.
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

        # Let's now send some traffic to the hosted model. We'll repeat
        # the process until we've sent the specified number of samples.
        while self.dispatched_samples < self.samples:
            # We want to send traffic in batches (instead of sending
            # samples one by one). Let's make sure we don't go over
            # the number of specified samples.
            batch = min(self.samples - self.dispatched_samples, 10)

            batch = self.data.sample(n=batch)

            payload = [
                {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
                for _, row in batch.iterrows()
            ]

            # Now that we have the payload we want to send with a
            # batch of samples, we can use the Backend implementation
            # to invoke the hosted model.
            predictions = self.backend_impl.invoke(payload)
            if predictions is None:
                self.logger().error("Failed to get predictions from the hosted model.")
                break

            self.dispatched_samples += len(batch)

        self.next(self.end)

    @step
    def end(self):
        """End of the pipeline."""
        self.logger.info(
            "Sent %s samples to the hosted model.", self.dispatched_samples
        )


if __name__ == "__main__":
    Traffic()
