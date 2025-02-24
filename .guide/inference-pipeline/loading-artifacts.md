# Loading Artifacts

After [initializing the backend](.guide/inference-pipeline/initializing-backend.md) as part of the [`load_context`](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PythonModel.load_context) function, the pipeline will load the artifacts from the context object.

When we registered the model, we packaged the two Scikit-Learn transformation pipelines and the Keras model together:

```python 
artifacts = {
    "model": model_path,
    "features_transformer": features_transformer_path,
    "target_transformer": target_transformer_path,
}
```

We can now load these artifacts in memory to use them in the inference process:

```python 
features_transformer = joblib.load(context.artifacts["features_transformer"])
target_transformer = joblib.load(context.artifacts["target_transformer"])
model = keras.saving.load_model(context.artifacts["model"])
```

Since the Training pipeline used a TensorFlow backend to train the model, we need to initialize the `KERAS_BACKEND` environment variable with `tensorflow` before loading the model from the artifacts.

You can run the [tests](tests/model/test_model_artifacts.py) associated with loading the artifacts by executing the following command:

```shell
uv run -- pytest -k test_model_artifacts
```
