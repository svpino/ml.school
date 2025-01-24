# Serving The Model

To deploy the model locally, we can use the `mflow models serve` command specifying the model version we want to deploy from the model registry. You can find more information about local deployments in [Deploy MLflow Model as a Local Inference Server](https://mlflow.org/docs/latest/deployment/deploy-model-locally.html).

The command below starts a local server listening in port `8080`. This server will host the latest version of the model from the model registry:

```shell
just serve
```

You can see the actual command behind the `serve` recipe by opening the [`justfile`](/justfile) file. Notice how the command uses the `MLFLOW_TRACKING_URI` environment variable to get the latest version of the model from the model registry. Review the [Environment variables](.guide/introduction/env.md) section to learn more about the environment variables used in the project. 

If we want the model to capture the input data and the predictions it generates, we must specify a backend implementation using the `MODEL_BACKEND` environment variable. You can do that by running the following command:

```shell
MODEL_BACKEND=backend.Local just serve
```

The command above will use the `backend.Local` implementation and will capture the data in a SQLite database. You can also export the `MODEL_BACKEND` environment variable in your shell to avoid specifying it every time you run the command:

```shell
export MODEL_BACKEND=backend.Local
just serve
```

By default, MLflow uses [Flask](https://flask.palletsprojects.com/en/1.1.x/) to serve the inference endpoint. Flask is a lightweight web framework and might not be suitable for production use cases. If you need a more robust and scalable inference server, you can use [MLServer](https://mlserver.readthedocs.io/en/latest/), an open-source project that provides a standardized interface for deploying and serving models.

To deploy the model using MLServer, execute the `mlflow models serve` command above with the `--enable-mlserver` option.
