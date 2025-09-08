# Cross-Validation Strategy

The [Training pipeline](src/pipelines/training.py) uses a cross-validation process to evaluate the model's performance and then train a final model using the entire dataset.

We'll use 5-fold cross-validation to train and evaluate five separate models and average their accuracy to determine the final evaluation score of the system.

![Cross-validation](.guide/training-pipeline/images/cross-validation.png)

By splitting the dataset into multiple folds and training and evaluating on different splits, we reduce the risk of overestimating or underestimating performance compared to using a single train-test split.

In the `Training.cross_validation` step, we use a Metaflow [`foreach`](.guide/introduction-to-metaflow/foreach.md) clause to create an independent branch for each one of the folds:

```python
@step
def cross_validation(self):
    from sklearn.model_selection import KFold

    kfold = KFold(n_splits=5, shuffle=True)
    self.folds = list(enumerate(kfold.split(self.data)))
    self.next(self.transform_fold, foreach="folds")
```

Each branch will transform the data, train a model, and evaluate it.

At the end of the cross-validation process, we'll join these separate branches and compute the average score using the individual evaluation of each model. This will happen in the `average_scores` step.

You can run the [tests](tests/pipelines/test_training_cross_validation.py) associated with the cross-validation process by executing the following command:

```shell
uv run pytest -k test_training_cross_validation
```