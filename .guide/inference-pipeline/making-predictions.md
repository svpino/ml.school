# Making Predictions

The core of the inference pipeline happens in the `predict()` function. This method completes five steps to make a prediction using the request sent by the client.

First, it'll convert the request data to a Pandas `DataFrame` object so we can use it with the Scikit-Learn transformation pipelines:

```python
model_input = pd.DataFrame([sample.model_dump() for sample in model_input])
```

The second step transforms the input data using one of the Scikit-Learn transformation pipelines we loaded in the [`load_context()`](.guide/inference-pipeline/loading-artifacts.md) function. This step ensures we show the model clean data in the appropriate format:

```python
result = self.features_transformer.transform(payload)
```

With the transformed input data ready, the third step is to make a prediction using the Keras model:

```python
predictions = self.model.predict(transformed_payload)
```

The format of the prediction output is a softmax vector containing the probability of each of the three classes. Since we don't want to return this vector directly to the client, the fourth step will transform it into a readable JSON object that will look like this:

```json
[{
    "prediction": "Adelie",
    "confidence": 0.75
}]
```

The appropriate target name (`Adelie`, `Gentoo`, or `Chinstrap`) comes from the Scikit-Learn transformation pipeline that we used to transform the target column of the dataset. The `confidence` is the corresponding softmax value from the prediction vector.

Finally, the pipeline will capture the input request and the output prediction using the supplied backend implementation:

```python
if self.backend is not None:
    self.backend.save(model_input, model_output)
```

Notice that the inference pipeline will only capture the data if there's an active backend implementation. 

You can run the [tests](tests/model/test_model_predict.py) associated with loading the artifacts by executing the following command:

```shell
uv run -- pytest -k test_model_predict
```
