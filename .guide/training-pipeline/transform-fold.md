# Transforming Fold Data

The first step of the cross-validation process is to transform the data so we can use it to train a model.

The transformers use [Scikit-Learn pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) to orchestrate the transformation steps. They will impute missing values, scale numerical columns, and encode categorical features.

We'll use two transformers to preprocess the data: the first will transform the feature columns, and the second will transform the target column. You'll find the implementation of `build_features_transformer` and `build_target_transformer` in the [`common.py`](pipelines/common.py) file.

The `cross-validation` step generates the set of indices representing the training and test data. Since we are running a 5-fold validation strategy, we'll receive 80% of the data for training and the remaining 20% for testing. We can access these indices through Metaflow's `self.input` attribute:

```python
self.fold, (self.train_indices, self.test_indices) = self.input
```

We can use these indices to split the data into training and test sets, build the transformers, fit them on the training data, and transform the training and test data.

Pay attention to how we use [`fit_transform`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline.fit_transform) to fit the transformer on the train data before transforming it. We don't do the same with the test data:

```python
self.x_train = features_transformer.fit_transform(train_data)
self.x_test = features_transformer.transform(test_data)
```

In Scikit-learn, [`fit_transform`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline.fit_transform) computes the transformation parameters (like the column's mean and standard deviation for scaling) from the training data and applies the transformation in a single step. We want to use this approach **only on the training data**. In contrast, [`transform`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline.transform) uses the parameters already computed during [`fit`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline.fit) or [`fit_transform`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline.fit_transform) to transform the test data without recalculating those parameters. This approach ensures consistent preprocessing across training and unseen data.

We want to store the preprocessed data as artifacts in the flow because future steps will need access to this information. 

You can run the [tests](tests/test_training_transform.py) associated with the transformation process by executing the following command:

```shell
uv run -- pytest -k test_training_transform
```