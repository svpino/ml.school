# Integrating with MLflow

We want to integrate the Training pipeline with MLflow to track the training, evaluation, and registration processes.

The first step is to connect our local client to the tracking server using the [`mlflow.set_tracking_uri()`](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri) function. The `MLFLOW_TRACKING_URI` environment variable should be pointing to the tracking server so we can use it as the default value of the `mlflow_tracking_uri` parameter to configure the MLflow client. If the `MLFLOW_TRACKING_URI` environment variable hasn't been set, we will assume the tracking server is running locally.

The `mlflow_tracking_uri` parameter will also be useful in cases where we want to point the pipeline to a specific tracking server regardless of the value of the `MLFLOW_TRACKING_URI` environment variable. For example, to test the pipeline, we can set the `mlflow_tracking_uri` parameter to a temporal folder and skip logging test runs to the production tracking server:

```shell
uv run -- python pipelines/training.py \
    --with retry \
    --environment conda run \
    --mlflow-tracking-uri file:///tmp/mlflow
```

Finally, we want to start a new MLflow run every time the Training pipeline runs to keep related metrics and parameters organized. By default, MLflow generates a random run name every time the pipeline runs, but we can use the Metaflow run ID to connect the pipeline run with its corresponding MLflow run.

```python
run = mlflow.start_run(run_name=current.run_id)
```

By naming the MLflow run with the Metaflow run identifier, we can easily recognize how they relate to each other.

We want every pipeline step to contribute to the same MLflow run, so we will store the run identifier as a Metaflow artifact and use it whenever we need to connect to MLflow.
