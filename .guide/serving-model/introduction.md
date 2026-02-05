# Serving The Model

To deploy the model locally, we can use the `mlflow models serve` command specifying the model version we want to deploy from the model registry. You can find more information about local deployments in [Deploy MLflow Model as a Local Inference Server](https://mlflow.org/docs/latest/deployment/deploy-model-locally.html).

Run the following command to start a local inference server listening on port `8080`. This server will host the latest version of the model from the model registry:

```shell
just serve
```

You can see the actual command details by opening the [`justfile`](/justfile) file. Notice how the command uses the `MLFLOW_TRACKING_URI` environment variable to retrieve the latest version of the model from the model registry. 

By default, we'll serve the model using the `backend.Local` implementation. This implementation will save requests and predictions to a local SQLite database. This is useful for development and testing purposes. 

Run the following command to query the local SQLite database used by the `backend.Local` implementation and see how many samples have been processed by the inference server:

```shell
just sqlite
```

For production use cases, you might want to use a different backend. To specify a different implementation, you can set the `MODEL_BACKEND` environment variable.

By default, MLflow uses [Flask](https://flask.palletsprojects.com/en/1.1.x/) to serve the inference endpoint. Flask is a lightweight web framework and might not be suitable for production use cases. If you need a more robust and scalable inference server, you can use [MLServer](https://mlserver.readthedocs.io/en/latest/), an open-source project that provides a standardized interface for deploying and serving models.