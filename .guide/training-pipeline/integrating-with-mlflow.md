# Integrating with MLflow

We want to integrate the [Training pipeline](pipelines/training.py) with MLflow to track the training, evaluation, and registration of the model.

To integrate with MLflow, we need to connect the local client to the tracking server using the [`mlflow.set_tracking_uri()`](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri) function. We need to call this function in every flow step before connecting to MLflow.

We can implement a custom `@mlflow` [decorator](.guide/introduction-to-metaflow/decorators-and-mutators.md) to simplify this process. The decorator will automatically set the tracking URI to the value of the `Pipeline.mlflow_tracking_uri` parameter. Using a [mutator](.guide/introduction-to-metaflow/decorators-and-mutators.md), we can automatically apply the `@mlflow` decorator to every step of the flow.

You'll find the implementation of the `@mlflow` decorator and the `pipeline` mutator in the [common.py](pipelines/common.py) file.

The `Pipeline` base class implements the `mlflow-tracking-uri` property that points to the URI of the MLflow tracking server. If it's not specified, this property will use the value of the  `MLFLOW_TRACKING_URI` environment variable by default. If the environment variable hasn't been set, we will assume the tracking server is running locally.

This `mlflow-tracking-uri` property will be useful in cases where we want to point the pipeline to a specific tracking server regardless of the value of the `MLFLOW_TRACKING_URI` environment variable. For example, to test the pipeline, we can set the `mlflow-tracking-uri` parameter to a temporal folder and skip logging test runs to the production tracking server:

```shell
uv run pipelines/training.py \
    --with retry run \ 
    --mlflow-tracking-uri file:///tmp/mlflow
```

When the [Training pipeline](pipelines/training.py) starts, we want to start a new MLflow run to keep related metrics and parameters organized. By default, MLflow generates a random run name every time the pipeline runs, but we can use the Metaflow run ID (`current.run_id`) to connect the pipeline run with its corresponding MLflow run:

```python
run = mlflow.start_run(run_name=current.run_id)
```

By naming the MLflow run with the Metaflow run identifier, we can easily recognize how they relate to each other.

We want every pipeline step to contribute to the same MLflow run, so we will store the run identifier as a Metaflow artifact and use it whenever we need to connect to MLflow.
