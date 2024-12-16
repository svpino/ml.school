# Integrating with MLflow

We want to integrate the Training pipeline with MLflow to track the entire training and evaluation workflow and register the model at the end of it.

The first step is to connect our local client to the tracking server using the [`mlflow.set_tracking_uri()`](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri) function. The `MLFLOW_TRACKING_URI` environment variable should be pointing to the tracking server so we can load it on the pipeline's `start` step and use it to configure the MLflow client. If the `MLFLOW_TRACKING_URI` environment variable hasn't been set, we will assume the tracking server is running locally.

Notice that we need to configure the local client on every pipeline step that uses it because Metaflow steps may run on separate computing instances. To simplify this, we can make the value of the `MLFLOW_TRACKING_URI` environment variable available at every step through a Metaflow artifact.

Finally, we want to start a new MLflow run every time the Training pipeline runs to keep related metrics and parameters organized. By default, MLflow generates a random run name every time the pipeline runs, but we can use the Metaflow run ID to connect the pipeline run with its corresponding MLflow run.

```python
run = mlflow.start_run(run_name=current.run_id)
```

By naming the MLflow run with the Metaflow run identifier, we can easily see which experiment corresponds to a particular pipeline run.

We want every pipeline step to contribute to the same MLflow run, so we will store the run identifier as a Metaflow artifact and use it whenever we need to connect to MLflow.
