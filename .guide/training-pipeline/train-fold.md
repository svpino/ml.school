# Training the Model

We'll use [Keras](https://keras.io/) with a JAX backend to train the model. You can swap to a different backend by setting the `KERAS_BACKEND` environment variable to any supported backends, such as `torch` or `tensorflow`.

To ensure the `KERAS_BACKEND` environment variable is available in the `train` step, we'll use the Metaflow [`@environment`](.guide/introduction-to-metaflow/environment.md) decorator. If the environment variable doesn't exist, the decorator will create and initialize it to `jax`.

To track the training process of each of the folds of our cross-validation strategy, we'll create a nested MLflow run to track each fold individually. We'll name this run using the index of the current fold so we can easily differentiate them.

We want to track the training process under the same MLflow run we started at the beginning of the flow. Since we are running cross-validation, we can create a nested run for each fold to keep track of each model individually:

```python
with (
    mlflow.start_run(run_id=self.mlflow_run_id),
    mlflow.start_run(
        run_name=f"cross-validation-fold-{self.fold}", 
        nested=True
    ) as run,
):
    ...
```

To avoid registering the individual cross-validation models, we can turn off MLflow's auto-logging functionality. Remember, we only want to register the final model at the end of the training process:

```python
mlflow.autolog(log_models=False)
```

We are going to build a simple neural network to solve this problem. You could use several different algorithms to build this model (I always recommend starting with tree-based models for tabular data problems,) but a simple neural network works just fine. You'll find the implementation of `build_model` in the [`common.py`](pipelines/common.py) file.

Here is the architecture of the neural network:

[IMAGE]

After building the model, we can fit it using the training data we preprocessed in the previous pipeline step. Notice how we use the `training_epochs` and `training_batch_size` properties to control the training process. You can experiment with different values when running the pipeline:

```shell
uv run -- python pipelines/training.py \
    --environment=conda run \
    --training-epochs 10 \
    --training-batch-size 16
```
You can run the [tests](tests/test_common_model.py) associated with building the model by executing the following command:

```shell
uv run -- pytest -k test_common_model \
    -W ignore::DeprecationWarning
```



