import logging
import os

from common import PYTHON, DatasetMixin, configure_logging, packages
from inference.endpoint import EndpointMixin
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
class Deployment(FlowSpec, DatasetMixin, EndpointMixin):
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

        self.endpoint_impl = self.load_endpoint()
        self.data = self.load_dataset()
        self.latest_model = self._get_latest_model_from_registry()

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
            logging.info("Model artifacts downloaded to %s ", self.model_artifacts)

            self.endpoint_impl.deploy(
                self.model_artifacts,
                self.latest_model.version,
            )

        self.next(self.inference)

    @step
    def inference(self):
        """Run a few samples through the deployed model to make sure it's working."""
        # Let's select a few random samples from the dataset.
        samples = self.data.sample(n=3).drop(columns=["species"]).reset_index(drop=True)

        self.endpoint_impl.invoke(samples)
        self.next(self.end)

    @step
    def end(self):
        """Finalize the deployment pipeline."""
        logging.info("The End")

    def _get_latest_model_from_registry(self):
        """Get the latest model version from the model registry."""
        from mlflow import MlflowClient

        logging.info("Loading the latest model version from the model registry...")

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

    def _deploy_to_azure(self):
        """Deploy the model to Azure ML.

        This function creates a new Azure model, endpoint, and deployment to serve the
        latest version of the model.

        If the endpoint already exists and there's an active deployment associated
        with it, this function will create a new deployment, route 100% of the traffic
        to it, and delete the previous deployment.
        """
        import os

        import mlflow
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

        # Let's start by getting the configuration to connect to Azure from
        # environment variables.
        subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
        resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
        workspace = os.environ.get("AZURE_WORKSPACE")

        if not all([subscription_id, resource_group, workspace]):
            message = (
                "Missing required environment variables. "
                "To deploy the model to Azure, you need to set the "
                "AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, and AZURE_WORKSPACE "
                "environment variables."
            )
            raise RuntimeError(message)

        # Let's connect to Azure and get the tracking URI that we need to configure
        # MLflow to use the Azure ML workspace.
        ml_client = MLClient(
            DefaultAzureCredential(),
            subscription_id,
            resource_group,
            workspace,
        )

        self.deployment_target_uri = ml_client.workspaces.get(
            ml_client.workspace_name,
        ).mlflow_tracking_uri

        mlflow.set_tracking_uri(self.deployment_target_uri)

        model = self._create_azure_model()
        self._create_azure_endpoint()
        self._create_azure_deployment(model)

    def _create_azure_model(self):
        """Create an Azure model if it doesn't exist.

        The first step to deploy a model to Azure is to register it. This function will
        register the model version if it doesn't exist. Azure will automatically assign
        a new version number to the model, so we'll keep the original version number as
        a tag to keep track of it.
        """
        from mlflow import MlflowClient

        model_name = "penguins"

        # Let's connect to Azure and return every model that matches the name we're
        # going to use to register the model.
        mlflow_client = MlflowClient(self.deployment_target_uri)
        models = mlflow_client.search_model_versions(
            filter_string=f"name = '{model_name}'",
        )

        # If we find any matching models, we need to check whether the latest version
        # we're going to use is already registered. Notice how we're using the version
        # number stored as a tag.
        model = next(
            (
                m
                for m in models
                if int(m.tags.get("version", 0)) == self.latest_model.version
            ),
            None,
        )

        if model:
            logging.info('Model "%s" already exists.', model_name)
            return model

        # If we don't find a model that matches the latest version, we can register
        # the model in Azure.
        logging.info('Creating model "%s"...', model_name)
        mlflow_client.create_registered_model(model_name)
        return mlflow_client.create_model_version(
            name=model_name,
            source=self.model_artifacts,
            # We want to store the model version as a tag.
            tags={"version": self.latest_model.version},
        )

    def _create_azure_endpoint(self):
        """Create an Azure endpoint if it doesn't exist.

        An endpoint is the entry point that clients will use for online (real-time)
        inferencing. This function will create the endpoint if it doesn't exist.
        """
        from azure.core.exceptions import ResourceNotFoundError
        from mlflow.deployments import get_deploy_client

        deployment_client = get_deploy_client(self.deployment_target_uri)

        try:
            # Let's try to get the endpoint. If it doesn't exist, this function will
            # raise an exception.
            deployment_client.get_endpoint(self.endpoint)
            logging.info('Endpoint "%s" already exists.', self.endpoint)
        except ResourceNotFoundError:
            logging.info('Creating endpoint "%s"...', self.endpoint)
            deployment_client.create_endpoint(self.endpoint)

    def _create_azure_deployment(self, model):
        """Create an Azure deployment if it doesn't exist.

        A deployment is the set of resources required for hosting the model behind an
        endpoint. This function will create a new deployment if it doesn't exist, route
        all traffic to it, and delete the previous deployment.
        """
        import json
        import tempfile

        from mlflow.deployments import get_deploy_client

        # Let's setup the name of the deployment we want to create. We want to store
        # this name as an artifact of the flow to use it later to make predictions.
        self.deployment_name = f"{self.endpoint}-{self.latest_model.version}"

        deployment_client = get_deploy_client(self.deployment_target_uri)

        # Let's get the list of deployments associated with the endpoint.
        deployments = deployment_client.list_deployments(self.endpoint)

        # We don't want to do anything if the deployment already exists, so let's
        # display a message and leave.
        if any(d["name"] == self.deployment_name for d in deployments):
            logging.info('Deployment "%s" already exists.', self.deployment_name)
            return

        # If we need to create a new deployment, let's store the name of the current
        # deployment so we can delete it later.
        previous_deployment = deployments[0]["name"] if len(deployments) > 0 else None

        # To configure the deployment and its traffic, we need to create two temporary
        # configuration files with the settings we want to give to Azure. I don't like
        # this but this is how their SDK works.
        with (
            tempfile.NamedTemporaryFile(mode="w") as deployment_config,
            tempfile.NamedTemporaryFile(mode="w") as traffic_config,
        ):
            # We are going to use a single instance to host the model.
            json.dump(
                {
                    "instance_type": "Standard_DS3_v2",
                    "instance_count": 1,
                },
                deployment_config,
            )

            # We want to route 100% of the traffic to the new deployment. If you wanted
            # to implement a staged rollout, you would configure the traffic
            # distribution between deployments here.
            json.dump(
                {
                    "traffic": {
                        self.deployment_name: 100,
                    },
                },
                traffic_config,
            )

            # Let's flush the configuration files to disk so we can use them.
            deployment_config.flush()
            traffic_config.flush()

            # Now we can create the new deployment using the current model.
            logging.info('Creating new deployment "%s"...', self.deployment_name)
            deployment_client.create_deployment(
                name=self.deployment_name,
                endpoint=self.endpoint,
                model_uri=f"models:/{model.name}/{model.version}",
                config={"deploy-config-file": deployment_config.name},
            )

            # After creating the deployment, we need to update the traffic distribution
            # to route all traffic to it.
            logging.info("Updating endpoint traffic...")
            deployment_client.update_endpoint(
                endpoint=self.endpoint,
                config={"endpoint-config-file": traffic_config.name},
            )

            # Finally, if there was a previous active deployment, we need to delete it.
            if previous_deployment:
                logging.info(
                    'Deleting previous deployment "%s"...',
                    previous_deployment,
                )
                deployment_client.delete_deployment(
                    name=previous_deployment,
                    endpoint=self.endpoint,
                )

    def _run_azure_prediction(self, samples):
        from mlflow.deployments import get_deploy_client

        deployment_client = get_deploy_client(self.deployment_target_uri)

        logging.info(
            'Running prediction on "%s/%s"...',
            self.endpoint,
            self.deployment_name,
        )

        response = deployment_client.predict(
            endpoint=self.endpoint,
            deployment_name=self.deployment_name,
            df=samples,
        )

        logging.info("\n%s", response)


if __name__ == "__main__":
    Deployment()
