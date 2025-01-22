# Initializing The Backend

As soon as MLflow creates the model, it will call the [`load_context`](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PythonModel.load_context) function, which we can use to load the artifacts the inference pipeline needs and configure it.

MLflow will call `load_context` only once when the model is loaded, so it's the perfect time to load any files in memory or perform any time-consuming initialization operations.

The first step is to initialize the backend instance that the pipeline will use to store the input requests and the predictions from the model. Since we want to deploy the model to different platforms and keep it as flexible as possible, we'll dynamically create an instance of the backend that will allow us to store the data.

The custom model relies on the `MODEL_BACKEND` environment variable to determine which class it should dynamically load to store the data. By default, the pipeline will not store the inputs and predictions. 

One of the backend implementations included as part of the code available to the pipeline is a Local backend that stores the data in a SQLite database. You can use this backend by setting the `MODEL_BACKEND` environment variable to `backend.Local` in the environment where the model is running.

If `MODEL_BACKEND` is specified, the pipeline will create and initialize an instance of the class:

```python
module, cls = backend_class.rsplit(".", 1)
module = importlib.import_module(module)
backend = getattr(module, cls)(config=...)
```

If the `MODEL_BACKEND_CONFIG` environment variable is specified, the pipeline will attempt to load it as a JSON file and pass a dictionary of settings to the backend implementation for initialization.

You can run the [tests](tests/model/test_model_backend.py) associated with initializing the backend by executing the following command:

```shell
uv run -- pytest -k test_model_backend
```
