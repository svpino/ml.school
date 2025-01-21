import importlib
import json
import logging
import os
import re
from abc import ABC, abstractmethod

import pandas as pd
import requests
from metaflow import Config, Parameter
from mlflow.deployments import get_deploy_client
from mlflow.exceptions import MlflowException
from sagemaker import get_boto3_client


class EndpointMixin:
    """A mixin for managing the endpoint implementation.

    This mixin is designed to be combined with any pipeline that requires sending
    traffic to a hosted model.
    """

    endpoint_config = Config(
        "endpoint",
        help=("Endpoint configuration used to initialize the provided endpoint class."),
    )

    endpoint = Parameter(
        "endpoint",
        help=(
            "Class implementing the `endpoint.Endpoint` abstract class. "
            "This class is responsible making predictions using a hosted model."
        ),
        default="endpoint.Server",
    )

    def load_endpoint(self):
        """Instantiate the endpoint class using the supplied configuration."""
        try:
            module, cls = self.endpoint.rsplit(".", 1)
            module = importlib.import_module(module)
            endpoint_impl = getattr(module, cls)(config=self._get_config())
        except Exception as e:
            message = f"There was an error instantiating class {self.endpoint}."
            raise RuntimeError(message) from e
        else:
            logging.info("Endpoint: %s", self.endpoint)
            return endpoint_impl

    def _get_config(self):
        """Return the endpoint configuration with environment variables expanded."""
        if not self.endpoint_config:
            return None

        config = self.endpoint_config.to_dict()
        pattern = re.compile(r"\$\{(\w+)\}")

        def replacer(match):
            env_var = match.group(1)
            return os.getenv(env_var, f"${{{env_var}}}")

        for key, value in self.endpoint_config.items():
            if isinstance(value, str):
                config[key] = pattern.sub(replacer, value)

        return config


class Endpoint(ABC):
    """Interface for making predictions using a hosted model."""

    @abstractmethod
    def invoke(self, payload: dict) -> dict | None:
        """Make a prediction request to the hosted model."""

    @abstractmethod
    def deploy(self, model_artifacts: str, latest_model_version: str) -> None:
        """Deploy the model to an endpoint.

        Args:
            model_artifacts: The location of the model artifacts.
            latest_model_version: The version of the model to deploy.

        """


class Server(Endpoint):
    """A class for making predictions using an inference server."""

    def __init__(self, target: str, **kwargs) -> None:
        """Initialize the class with the target location of the model."""
        self.target = target

    def invoke(self, payload: dict) -> dict | None:
        """Make a prediction request to the hosted model."""
        logging.info('Running prediction on "%s"...', self.target)

        try:
            predictions = requests.post(
                url=self.target,
                headers={"Content-Type": "application/json"},
                data=json.dumps(
                    {
                        "inputs": payload,
                    },
                ),
                timeout=5,
            )
            return predictions.json()
        except Exception:
            logging.exception("There was an error sending traffic to the endpoint.")
            return None

    def deploy(self, model_artifacts: str, latest_model_version: str) -> None:
        """Deploying a model is not applicable when serving the model directly."""


class Sagemaker(Endpoint):
    """Sagemaker endpoint implementation."""

    def __init__(self, config: dict | None = None) -> None:
        """Initialize the endpoint instance."""
        self.target = config.get("target", "penguins") if config else "penguins"
        self.data_capture_destination = (
            config.get("data-capture-destination", None) if config else None
        )
        self.assume_role = config.get("assume_role", None) if config else None
        self.region = config.get("region", "us-east-1") if config else "us-east-1"

        self.deployment_target_uri = (
            f"sagemaker:/{self.region}/{self.assume_role}"
            if self.assume_role
            else f"sagemaker:/{self.region}"
        )

        self.deployment_client = get_deploy_client(self.deployment_target_uri)

        logging.info("Target: %s", self.target)
        logging.info("Data capture destination: %s", self.data_capture_destination)
        logging.info("Assume role: %s", self.assume_role)
        logging.info("Region: %s", self.region)
        logging.info("Deployment target URI: %s", self.deployment_target_uri)

    def invoke(self, payload: dict) -> dict | None:
        """Make a prediction request to the Sagemaker endpoint."""
        logging.info('Running prediction on "%s"...', self.target)

        response = self.deployment_client.predict(self.target, payload)
        df = pd.DataFrame(response["predictions"])[["prediction", "confidence"]]

        logging.info("\n%s", df)

        return df.to_json()

    def deploy(self, model_artifacts: str, latest_model_version: str) -> None:
        """Deploy the model to SageMaker.

        This function creates a new Sagemaker Model, Sagemaker Endpoint Configuration,
        and Sagemaker Endpoint to serve the latest version of the model.

        If the endpoint already exists, this function will update it with the latest
        version of the model.
        """
        deployment_configuration = {
            "instance_type": "ml.m4.xlarge",
            "instance_count": 1,
            "synchronous": True,
            # We want to archive resources associated with the endpoint that become
            # inactive as the result of updating an existing deployment.
            "archive": True,
            # Notice how we are storing the version number as a tag.
            "tags": {"version": latest_model_version},
        }

        # If the data capture destination is defined, we can configure the SageMaker
        # endpoint to capture data.
        if self.data_capture_destination is not None:
            deployment_configuration["data_capture_config"] = {
                "EnableCapture": True,
                "InitialSamplingPercentage": 100,
                "DestinationS3Uri": self.data_capture_destination,
                "CaptureOptions": [
                    {"CaptureMode": "Input"},
                    {"CaptureMode": "Output"},
                ],
                "CaptureContentTypeHeader": {
                    "CsvContentTypes": ["text/csv", "application/octect-stream"],
                    "JsonContentTypes": [
                        "application/json",
                        "application/octect-stream",
                    ],
                },
            }

        if self.assume_role:
            deployment_configuration["execution_role_arn"] = self.assume_role

        try:
            # Let's return the deployment with the name of the endpoint we want to
            # create. If the endpoint doesn't exist, this function will raise an
            # exception.
            deployment = self.deployment_client.get_deployment(self.target)

            # We now need to check whether the model we want to deploy is already
            # associated with the endpoint.
            if self._is_sagemaker_model_running(deployment, latest_model_version):
                logging.info(
                    'Enpoint "%s" is already running model "%s".',
                    self.target,
                    latest_model_version,
                )
            else:
                # If the model we want to deploy is not associated with the endpoint,
                # we need to update the current deployment to replace the previous model
                # with the new one.
                self._update_sagemaker_deployment(
                    deployment_configuration,
                    model_artifacts,
                    latest_model_version,
                )
        except MlflowException:
            # If the endpoint doesn't exist, we can create a new deployment.
            self._create_sagemaker_deployment(
                deployment_configuration,
                model_artifacts,
                latest_model_version,
            )

    def _is_sagemaker_model_running(self, deployment, latest_model_version):
        """Check if the model is already running in SageMaker.

        This function will check if the current model is already associated with a
        running SageMaker endpoint.
        """
        sagemaker_client = get_boto3_client(
            service="sagemaker",
            assume_role=self.assume_role,
        )

        # Here, we're assuming there's only one production variant associated with
        # the endpoint. This code will need to be updated if an endpoint could have
        # multiple variants.
        variant = deployment.get("ProductionVariants", [])[0]

        # From the variant, we can get the ARN of the model associated with the
        # endpoint.
        model_arn = sagemaker_client.describe_model(
            ModelName=variant.get("VariantName"),
        ).get("ModelArn")

        # With the model ARN, we can get the tags associated with the model.
        tags = sagemaker_client.list_tags(ResourceArn=model_arn).get("Tags", [])

        # Finally, we can check whether the model has a `version` tag that matches
        # the model version we're trying to deploy.
        model = next(
            (
                tag["Value"]
                for tag in tags
                if (tag["Key"] == "version" and tag["Value"] == latest_model_version)
            ),
            None,
        )

        return model is not None

    def _create_sagemaker_deployment(
        self,
        deployment_configuration,
        model_artifacts,
        latest_model_version,
    ):
        """Create a new SageMaker deployment using the supplied configuration."""
        logging.info(
            'Creating endpoint "%s" with model "%s"...',
            self.target,
            latest_model_version,
        )

        self.deployment_client.create_deployment(
            name=self.target,
            model_uri=model_artifacts,
            flavor="python_function",
            config=deployment_configuration,
        )

    def _update_sagemaker_deployment(
        self,
        deployment_configuration,
        model_artifacts,
        latest_model_version,
    ):
        """Update an existing SageMaker deployment using the supplied configuration."""
        logging.info(
            'Updating endpoint "%s" with model "%s"...',
            self.target,
            latest_model_version,
        )

        # If you wanted to implement a staged rollout, you could extend the deployment
        # configuration with a `mode` parameter with the value
        # `mlflow.sagemaker.DEPLOYMENT_MODE_ADD` to create a new production variant. You
        # can then route some of the traffic to the new variant using the SageMaker SDK.
        self.deployment_client.update_deployment(
            name=self.target,
            model_uri=model_artifacts,
            flavor="python_function",
            config=deployment_configuration,
        )
