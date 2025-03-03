import logging
import os

from common import PYTHON, DatasetMixin, configure_logging, packages
from inference.backend import BackendMixin
from metaflow import (
    FlowSpec,
    conda_base,
    environment,
    project,
    step,
)

configure_logging()


@project(name="penguins")
@conda_base(
    python=PYTHON,
    packages=packages("mlflow", "boto3"),
)
class Deployment(FlowSpec, DatasetMixin, BackendMixin):
    """Deployment pipeline.

    This pipeline deploys the latest model from the model registry to a target platform
    and runs a few samples through the deployed model to ensure it's working.
    """

    @environment(
        vars={
            "MLFLOW_TRACKING_URI": os.getenv(
                "MLFLOW_TRACKING_URI",
                "http://127.0.0.1:5000",
            ),
        },
    )
    @step
    def start(self):
        """Start the deployment pipeline."""
        import mlflow

        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        logging.info("MLflow tracking URI: %s", self.mlflow_tracking_uri)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        print("Estoy en start. Backend config", self.config)

        self.backend_impl = self.load_backend()
        self.data = self.load_dataset()

        # TBD self.latest_model = self._get_latest_model_from_registry()

        self.next(self.deployment)

    @step
    def deployment(self):
        """Deploy the model to the appropriate target platform."""
        import tempfile
        from pathlib import Path

        import mlflow

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # Let's download the model artifacts from the model registry to a temporary
        # directory. This is the copy that we'll use to deploy the model.
        with tempfile.TemporaryDirectory() as directory:
            mlflow.artifacts.download_artifacts(
                run_id=self.latest_model.run_id,
                dst_path=directory,
            )

            self.model_artifacts = f"file://{(Path(directory) / 'model').as_posix()}"
            logging.info("Model artifacts downloaded to %s ",
                         self.model_artifacts)

            # TBD self.backend_impl.deploy(
            #     self.model_artifacts,
            #     self.latest_model.version,
            # )

        self.next(self.inference)

    @step
    def inference(self):
        """Run a few samples through the deployed model to make sure it's working."""
        # Let's select a few random samples from the dataset.
        samples = self.data.sample(n=3).drop(
            columns=["species"]).reset_index(drop=True)
        self.backend_impl.invoke(samples.to_dict(orient="records"))
        self.next(self.end)

    @step
    def end(self):
        """Finalize the deployment pipeline."""
        logging.info("The End")

    def _get_latest_model_from_registry(self):
        """Get the latest model version from the model registry."""
        from mlflow import MlflowClient

        logging.info(
            "Loading the latest model version from the model registry...")

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
        logging.info("Latest model version: %s", latest_model.version)
        logging.info("Latest model artifacts: %s.", latest_model.source)

        return latest_model


if __name__ == "__main__":
    Deployment()
