# Averaging Model Scores

As soon as we finish evaluating each of the models from the cross-validation process, we can compute the final performance by averaging the accuracy across all models.

While it might be tempting to report the accuracy of the best-performing model from the cross-validation process, this approach can lead to an overly optimistic performance estimate. The best model might reflect random variations in the data rather than genuine superiority.

Instead of averaging every score, you could remove the best and the worst-performing models before calculating the average. This approach will reduce the influence of outliers caused by random fluctuations or irregularities in specific folds. By focusing on the middle range of performance, you can get a more robust estimate of how the model will perform on unseen data.

The `average_scores` step acts as a [`join`](.guide/introduction-to-metaflow/branches.md) step where every cross-validation branch will converge. To propagate the value of the artifacts created earlier in the flow, we use the [`merge_artifacts()`](https://docs.metaflow.org/api/flowspec#FlowSpec.merge_artifacts) function. In this case, we want access to the `mlflow_run_id` to log the final model performance to MLflow:

```python
self.merge_artifacts(inputs, include=["mlflow_run_id"])
```

We can access the individual `test_accuracy` and `test_loss` values computed in each branch through the `inputs` property to average them. After we have the final values, we can store them as artifacts and log them to the tracking server using the identifier of the MLflow run.