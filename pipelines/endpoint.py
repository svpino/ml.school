import json
from abc import ABC, abstractmethod

import requests


class Endpoint(ABC):
    """TBD"""

    @abstractmethod
    def invoke(self, payload):
        """Invoke the model with a payload."""


class Local(Endpoint):
    def __init__(self, target: str) -> None:
        self.target = target

    def invoke(self, payload):
        """Invoke the model with a payload."""
        predictions = requests.post(
            url=self.target,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=5,
        )
        return predictions.json()


class Sagemaker(Endpoint):
    def __init__(self, config: dict | None = None) -> None:
        pass
