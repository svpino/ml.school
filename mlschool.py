import json

import boto3
import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mlschool")


@mcp.tool()
async def is_endpoint_running(endpoint: str) -> bool:
    """Check if a SageMaker endpoint is currently running.

    Args:
        endpoint: The name of the SageMaker endpoint to check

    Returns:
        bool: True if the endpoint is in service, False otherwise

    """
    try:
        sagemaker_client = boto3.client("sagemaker")
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint)

        return response["EndpointStatus"] == "InService"
    except Exception as e:
        print(f"Error checking endpoint status: {e}")
        return False


@mcp.tool()
async def invoke_model(payload: list[dict], url: str = "http://127.0.0.1:8080/invocations") -> str:
    """Invoke the hosted local model with the given payload.

    Args:
        payload: The payload that will be used to invoke the model. The payload is an
            array of dictionaries, each representing a sample. Here is an example
            payload:

            ```
            [
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
            ```

    Returns:
        str: The response from the model containing the predictions for each sample.

    """
    endpoint_url = "http://127.0.0.1:8080/invocations"
    headers = {"Content-Type": "application/json"}

    try:
        predictions = requests.post(
            url=endpoint_url,
            headers=headers,
            data=json.dumps(
                {
                    "inputs": payload,
                },
            ),
            timeout=5,
        )
        return predictions.json()

    except Exception as e:
        print(f"Error invoking model: {e}")
        return str(e)


def get_sample() -> list[dict]:
    """Get a sample payload for the model.

    Returns:
        list[dict]: A list of dictionaries, each representing a sample.

    """
    return [
        {"island": "Biscoe", "culmen_length_mm": 48.6, "culmen_depth_mm": 16.0,
            "flipper_length_mm": 230.0, "body_mass_g": 5800.0, "sex": "MALE"},
    ]


if __name__ == "__main__":
    mcp.run(transport="stdio")
