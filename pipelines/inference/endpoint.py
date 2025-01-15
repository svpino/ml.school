import importlib
import json
import logging
from abc import ABC, abstractmethod

import requests
from metaflow import Parameter


class EndpointMixin:
    """A mixin for managing the endpoint implementation.

    This mixin is designed to be combined with any pipeline that requires sending
    traffic to a hosted model.
    """

    endpoint = Parameter(
        "endpoint",
        help=(
            "Class implementing the `endpoint.Endpoint` abstract class. "
            "This class is responsible making predictions using a hosted model."
        ),
        default="endpoint.Server",
    )

    target = Parameter(
        "target",
        help=(
            "The location of the hosted model where the pipeline will send the traffic."
        ),
        default="http://127.0.0.1:8080/invocations",
    )

    def load_endpoint(self):
        """Instantiate the endpoint class using the supplied configuration."""
        try:
            module, cls = self.endpoint.rsplit(".", 1)
            module = importlib.import_module(module)
            endpoint_impl = getattr(module, cls)(target=self.target)
        except Exception as e:
            message = f"There was an error instantiating class {self.endpoint}."
            raise RuntimeError(message) from e
        else:
            logging.info("Endpoint: %s", self.endpoint)
            logging.info("Target: %s", self.target)
            return endpoint_impl


class Endpoint(ABC):
    """Interface for making predictions using a hosted model."""

    @abstractmethod
    def invoke(self, payload: dict) -> dict | None:
        """Make a prediction request to the hosted model."""


class Server(Endpoint):
    """A class for making predictions using an inference server."""

    def __init__(self, target: str) -> None:
        """Initialize the class with the target location of the model."""
        self.target = target

    def invoke(self, payload: dict) -> dict | None:
        """Make a prediction request to the hosted model."""
        try:
            predictions = requests.post(
                url=self.target,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=5,
            )
            return predictions.json()
        except Exception:
            logging.exception("There was an error sending traffic to the endpoint.")
            return None


class Sagemaker(Endpoint):
    def __init__(self, config: dict | None = None) -> None:
        pass
