import tempfile
from pathlib import Path

from metaflow import step

from common.pipeline import Pipeline, backend, dataset


class Deployment(Pipeline):
    """Deployment pipeline.

    This pipeline deploys the latest model from the model registry to a target platform
    and runs a few samples through the deployed model to ensure it's working.
    """

    @dataset
    @backend
    @step
    def start(self):
        """Start the deployment pipeline."""
        self.logger.info("MLflow tracking URI: %s", self.mlflow_tracking_uri)
        self.latest_model = self._get_latest_model_from_registry(self.logger)

        self.next(self.deployment)

    @step
    def deployment(self):
        """Deploy the model to the appropriate target platform."""
        import mlflow

        # Let's download the model artifacts from the model registry to a temporary
        # directory. This is the copy that we'll use to deploy the model.
        with tempfile.TemporaryDirectory() as directory:
            mlflow.pyfunc.load_model(
                model_uri=self.latest_model.source,
                dst_path=directory,
            )

            self.logger.info("Model artifacts downloaded to %s", directory)

            self.backend_impl.deploy(
                f"file://{Path(directory).as_posix()}",
                self.latest_model.version,
            )

        self.next(self.inference)

    @step
    def inference(self):
        """Run a few samples through the deployed model to make sure it's working."""
        # Let's select a few random samples from the dataset.
        samples = self.data.sample(n=3).drop(columns=["species"]).reset_index(drop=True)
        self.backend_impl.invoke(samples.to_dict(orient="records"))
        self.next(self.end)

    @step
    def end(self):
        """Finalize the deployment pipeline."""
        self.logger.info("The End")

    def _get_latest_model_from_registry(self, logger):
        """Get the latest model version from the model registry."""
        from mlflow import MlflowClient

        logger.info("Loading the latest model version from the model registry...")

        client = MlflowClient()
        response = client.search_model_versions(
            "name='penguins'",
            max_results=1,
            order_by=["last_updated_timestamp DESC"],
        )

        if not response:
            message = 'No model versions found registered under the name "penguins".'
            raise RuntimeError(message)

        latest_model = response[0]
        logger.info("Latest model version: %s", latest_model.version)
        logger.info("Latest model artifacts: %s.", latest_model.source)

        return latest_model


if __name__ == "__main__":
    Deployment()
