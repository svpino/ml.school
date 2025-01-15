import json
import logging
from abc import ABC, abstractmethod

import requests


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
