# Evaluating Fold Model 

After training the model for one of the cross-validation folds, we must evaluate it on the test data and store its performance in a Metaflow artifact we'll later use to compute the average score.

We'll use [Keras](https://keras.io/) to evaluate the model, so we need to ensure the `KERAS_BACKEND` environment variable is available in the evaluation step by using the [`@environment`](.guide/introduction-to-metaflow/environment.md) decorator. Notice how we evaluate the model using the test data, representing 20% of the dataset in a 5-fold cross-validation process:

```python
self.test_loss, self.test_accuracy = self.model.evaluate(
    self.x_test,
    self.y_test,
)
```

After computing the model's loss and accuracy on the test data, we can log it with MLflow under the run corresponding to the current cross-validation fold.

Finally, we can send the pipeline to the `average_scores` join step to compute the average scores across every one of the cross-validation models. Remember that this next step will only run after every parallel cross-validation branch finishes running.
