import logging
import sys

from metaflow import FlowSpec, Parameter, project, pypi_base, step

logger = logging.getLogger(__name__)


@project(name="penguins")
@pypi_base(
    python="3.10.14",
    packages={
        "mlflow": "2.15.1",
        "boto3": "1.35.8",
        "azure-ai-ml": "1.19.0",
        "azureml-mlflow": "1.57.0.post1",
    },
)
class DeploymentFlow(FlowSpec):
    from mlflow.sagemaker import DEPLOYMENT_MODE_REPLACE

    endpoint_name = Parameter(
        "endpoint_name",
        help=("The endpoint to deploy the model to"),
        default="penguins-endpoint",
    )

    target = Parameter(
        "target",
        help=("The target to deploy the model to"),
        default="sagemaker",
    )

    # TODO: Enforce that mode is one of the DEPLOYMENT_MODE_* constants
    mode = Parameter(
        "mode",
        help=("The mode to run the deployment in"),
        default=DEPLOYMENT_MODE_REPLACE,
    )

    @step
    def start(self):
        from mlflow import MlflowClient

        # TODO: What happens if there are no model versions?
        client = MlflowClient()
        self.latest_model = client.search_model_versions(
            "name='penguins'",
            max_results=1,
            order_by=["last_updated_timestamp DESC"],
        )[0]

        logger.info(
            "Latest model: %s. Source: %s",
            self.latest_model.version,
            self.latest_model.source,
        )

        self.next(self.deploy)

    @step
    def deploy(self):
        from mlflow.deployments import get_deploy_client
        from mlflow.exceptions import MlflowException

        if self.target == "sagemaker":
            client = get_deploy_client("sagemaker:/us-east-1")

            try:
                running_models = self._get_running_models(client)
                logger.info("Running models: %s", running_models)

                if self.latest_model.version not in running_models:
                    self._update_deployment(client)
                else:
                    logger.info(
                        "Enpoint is currently running model %s",
                        self.latest_model.version,
                    )

            except MlflowException:
                # self._create_deployment(client)
                pass
        elif self.target == "azure":
            import mlflow
            from azure.ai.ml import MLClient
            from azure.identity import DefaultAzureCredential
            from mlflow import MlflowClient

            subscription_id = "4cb304b6-6a53-464c-bd34-c65be6314534"
            resource_group = "mlflow"
            workspace = "main"

            ml_client = MLClient(
                DefaultAzureCredential(),
                subscription_id,
                resource_group,
                workspace,
            )

            self.azureml_tracking_uri = ml_client.workspaces.get(
                ml_client.workspace_name,
            ).mlflow_tracking_uri

            mlflow.set_tracking_uri(self.azureml_tracking_uri)
            mlflow_client = MlflowClient(tracking_uri=self.azureml_tracking_uri)

            model = self._create_azure_model(mlflow_client)
            self._create_azure_endpoint()
            self._create_azure_deployment(model)

            # if self._is_azure_model_registered(mlflow_client):
            #     logger.info(
            #         "Model %s is already registered",
            #         self.latest_model.version,
            #     )
            # else:
            #     self._deploy_azure_model(mlflow_client)

        self.next(self.end)

    @step
    def end(self):
        from mlflow.deployments import get_deploy_client

        if self.target == "sagemaker":
            payload1 = [
                {
                    "island": "Biscoe",
                    "culmen_length_mm": 48.6,
                    "culmen_depth_mm": 16.0,
                    "flipper_length_mm": 230.0,
                    "body_mass_g": 5800.0,
                    "sex": "MALE",
                },
                {
                    "island": "Biscoe",
                    "culmen_length_mm": 48.6,
                    "culmen_depth_mm": 16.0,
                    "flipper_length_mm": 230.0,
                    "body_mass_g": 5800.0,
                    "sex": "MALE",
                },
            ]

            client = get_deploy_client("sagemaker:/us-east-1/")
            response = client.predict(self.endpoint_name, payload1)
            print(response)
        elif self.target == "azure":
            import pandas as pd

            samples = (
                pd.read_csv("../penguins.csv")
                .sample(n=10)
                .drop(columns=["species"])
                .reset_index(drop=True)
            )

            deployment_client = get_deploy_client(self.azureml_tracking_uri)

            response = deployment_client.predict(
                endpoint=self.endpoint_name,
                df=samples,
            )

            print("FINAL RESPONSE", response)

    def _get_running_models(self, client):
        import boto3

        deployment = client.get_deployment(self.endpoint_name)

        models = []
        sagemaker_client = boto3.client("sagemaker")
        for variant in deployment.get("ProductionVariants", []):
            variant_name = variant.get("VariantName")
            model_arn = sagemaker_client.describe_model(ModelName=variant_name).get(
                "ModelArn",
            )
            tags = sagemaker_client.list_tags(ResourceArn=model_arn).get(
                "Tags",
                [],
            )
            model = next(
                (tag["Value"] for tag in tags if tag["Key"] == "model_version"),
                None,
            )

            models.append(int(model))

        return models

    def _create_azure_model(self, mlflow_client):
        model_name = "penguins"

        models = mlflow_client.search_model_versions(
            filter_string=f"name = '{model_name}'",
        )

        for model in models:
            version = model.tags.get("version")
            if version and int(version) == self.latest_model.version:
                return model

        return mlflow_client.create_model_version(
            name=model_name,
            source=self.latest_model.source,
            tags={"version": self.latest_model.version},
        )

    def _create_azure_endpoint(self):
        from azure.core.exceptions import ResourceNotFoundError
        from mlflow.deployments import get_deploy_client

        deployment_client = get_deploy_client(self.azureml_tracking_uri)

        try:
            endpoint = deployment_client.get_endpoint(self.endpoint_name)
        except ResourceNotFoundError:
            endpoint = deployment_client.create_endpoint(self.endpoint_name)

        return endpoint

    def _create_azure_deployment(self, model):
        from mlflow.deployments import get_deploy_client
        from mlflow.exceptions import MlflowException

        def create_endpoint_deployment(deployment_name):
            import json

            deploy_config = {
                "instance_type": "Standard_DS3_v2",
                "instance_count": 1,
            }

            deployment_config_path = "deployment_config.json"
            with open(deployment_config_path, "w") as outfile:
                outfile.write(json.dumps(deploy_config))

            deployment = deployment_client.create_deployment(
                name=deployment_name,
                endpoint=self.endpoint_name,
                model_uri=f"models:/{model.name}/{model.version}",
                config={"deploy-config-file": deployment_config_path},
            )

        def update_deployment_traffic(deployment_name, traffic):
            import json

            traffic_config = {"traffic": {deployment_name: traffic}}

            traffic_config_path = "traffic_config.json"
            with open(traffic_config_path, "w") as outfile:
                outfile.write(json.dumps(traffic_config))

            deployment_client.update_endpoint(
                endpoint=self.endpoint_name,
                config={"endpoint-config-file": traffic_config_path},
            )

        deployment_client = get_deploy_client(self.azureml_tracking_uri)

        deployment_name = f"{self.endpoint_name}-{self.latest_model.version}"
        try:
            deployment = deployment_client.get_deployment(
                endpoint=self.endpoint_name,
                name=deployment_name,
            )
        except MlflowException:
            create_endpoint_deployment(deployment_name)

        if self.mode == "replace":
            # Move traffic from old deployment to new deployment
            # Delete old deployment
            pass
        elif self.mode == "add":
            # Send 10% traffic to new deployment
            pass

    def _is_azure_model_registered(self, mlflow_client):
        model_versions = mlflow_client.search_model_versions(
            filter_string="name = 'penguins'",
        )

        models = []
        for model_version in model_versions:
            version = model_version.tags.get("version")
            if version:
                models.append(int(version))

        return self.latest_model.version in models

    def _deploy_azure_model(self, mlflow_client):
        import json

        from mlflow.deployments import get_deploy_client
        from mlflow.exceptions import MlflowException

        model_name = "penguins"

        model = mlflow_client.create_model_version(
            name=model_name,
            source=self.latest_model.source,
            tags={"version": self.latest_model.version},
        )

        deployment_client = get_deploy_client(self.azureml_tracking_uri)

        try:
            endpoint = deployment_client.get_endpoint(self.endpoint_name)
        except MlflowException:
            endpoint = deployment_client.create_endpoint(self.endpoint_name)

        deployment_name = "default"
        try:
            deployment = deployment_client.get_deployment(
                name=deployment_name,
                endpoint=self.endpoint_name,
            )
        except MlflowException:
            deploy_config = {
                "instance_type": "Standard_DS3_v2",
                "instance_count": 1,
            }

            deployment_config_path = "deployment_config.json"
            with open(deployment_config_path, "w") as outfile:
                outfile.write(json.dumps(deploy_config))

            deployment = deployment_client.create_deployment(
                name=deployment_name,
                endpoint=self.endpoint_name,
                model_uri=f"models:/{model_name}/{model.version}",
                config={"deploy-config-file": deployment_config_path},
            )

            traffic_config = {"traffic": {deployment_name: 100}}

            traffic_config_path = "traffic_config.json"
            with open(traffic_config_path, "w") as outfile:
                outfile.write(json.dumps(traffic_config))

            deployment_client.update_endpoint(
                endpoint=self.endpoint_name,
                config={"endpoint-config-file": traffic_config_path},
            )

    def _create_deployment(self, client):
        logger.info("Creating endpoint with model %s...", self.latest_model.version)

        # TODO: What happens if the deployment fails?
        client.create_deployment(
            name=self.endpoint_name,
            model_uri=self.latest_model.source,
            flavor="python_function",
            config={
                "instance_type": "ml.m4.xlarge",
                "instance_count": 1,
                "synchronous": True,
                "archive": True,
                "tags": {"model_version": self.latest_model.version},
            },
        )

    def _update_deployment(self, client):
        logger.info(
            'Updating endpoint with model %s [Mode "%s"]...',
            self.latest_model.version,
            self.mode,
        )

        client.update_deployment(
            name=self.endpoint_name,
            model_uri=self.latest_model.source,
            flavor="python_function",
            config={
                "instance_type": "ml.m4.xlarge",
                "instance_count": 1,
                "synchronous": True,
                "archive": True,
                "tags": {"model_version": self.latest_model.version},
                "mode": self.mode,
            },
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    DeploymentFlow()
