# Training the Final Model

After transforming the entire dataset, we can train the final model. We'll use the same architecture and hyperparameters we used during cross-validation.

To ensure the `KERAS_BACKEND` environment variable is available in the `train` step, we'll use the Metaflow [`@environment`](.guide/introduction-to-metaflow/environment.md) decorator. If the environment variable doesn't exist, the decorator will create and initialize it to `tensorflow`.

We'll log the training process in the tracking server under the current MLflow run. We don't want to automatically log the model because we'll do that during the registration process, so we'll turn off automatic logging:

```python
mlflow.autolog(log_models=False)
```

Notice that we'll store the model as a flow artifact to make it available for the registration step.

You can run the [tests](tests/test_training_train.py) associated with training the model by executing the following command:

```shell
uv run -- pytest -k test_training_train
```
