# Transforming the Dataset


We want to build the final model using the entire dataset, so the first step is to transform the data.

We can run the `transform` step parallel to the cross-validation process since there aren't any dependencies between these branches.

To transform the dataset, we'll use the same [Scikit-Learn pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) we used during cross-validation. These transformers will impute missing values, scale numerical columns, and encode categorical features. You'll find the pipeline implementation in the [`common.py`](pipelines/common.py) file.

Since we are training with the entire dataset, we don't need to set aside and transform any test data. The final model evaluation will come from [averaging the scores](.guide/training-pipeline/averaging-scores.md) after cross-validation.

We want to store the transformation pipelines as artifacts in the flow because we'll need to package them with the production model. When we deploy the model, we need to ensure it receives data in the same format as during training. We can achieve this by using the same transformations to process incoming data during inference.

You can run the [tests](tests/test_training_transform.py) associated with the transformation process by executing the following command:

```shell
uv run -- pytest -k test_training_transform
```