# Registering the Model

We are now ready to register the new model in the model registry, assuming its performance is above our predefined threshold.

First, the `register` step acts as a [`join`](.guide/introduction-to-metaflow/branches.md) step where the cross-validation branch converges with the branch that trains the model using the entire dataset. To propagate the value of the artifacts created earlier in the flow, we can use the [`merge_artifacts()`](https://docs.metaflow.org/api/flowspec#FlowSpec.merge_artifacts) function:

```python
self.merge_artifacts(inputs)
```

The Training pipeline has an `accuracy-threshold` parameter defining the minimum required performance for the model to make it into the registry. We don't want to register the model if the model's accuracy is under the threshold.

We'll use MLflow's [`python_function`](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html) model flavor to create and register the model. The MLflow model will be a wrapper around the specific model we trained. This wrapper will allow us to build an [inference pipeline](.guide/inference-pipeline/introduction.md) to validate and transform the input requests and outputs to and from the model. Check the [Custom MLflow Models with mlflow.pyfunc](https://mlflow.org/blog/custom-pyfunc) article for more information.

When deployed, our model needs access to the SciKit-Learn transformation pipelines to prepare the input data before running inference. We can package the fitted pipelines by saving them to a temporal directory and specifying the path to the files using the `artifacts` property of the model:

```python 
artifacts = {
    "model": model_path,
    "features_transformer": features_transformer_path,
    "target_transformer": target_transformer_path,
}
```

We also need to package any code files that will be necessary at inference time. The inference pipeline requires access to specific implementations of backend storage, so we'll include the [`inference/backend.py`](pipelines/inference/backend.py) file as part of the model package.

To ensure the model runs in production, we'll use the `pip_requirements` property to specify the list of required libraries. These libraries will be automatically installed by MLflow when preparing the container that will run the model.

Finally, we can register the model in the registry using the [`log_model`](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.log_model) function. The `python_model` property expects the path to the file defining the [`PythonModel`](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PythonModel) class that will act as the inference pipeline.

You can run the [tests](tests/test_training_register.py) associated with registering the model by executing the following command:

```shell
uv run -- pytest -k test_training_register
```
