# Assignments

Complete the following assignments to reinforce the concepts we covered in this session. You don't have to work on every assignment. Pick the ones that you think will help you the most.

1. The Training pipeline loads the dataset from a Comma-Separated Values (CSV) file that's specified when the pipeline runs. Modify the code, so when running in `production` mode, the pipeline loads the data from an S3 bucket. You can supply the S3 bucket name as a parameter to the pipeline. The code should load and combine every CSV file in that bucket.

1. The current pipeline loads the data from a single file. Modify the code to point the pipeline to a folder instead. The pipeline should load every file from the folder and combine them to create the dataset.

1. Write some code to determine whether the `sex` column has predictive power for the `species` column. In other words, do we have to include the `sex` column in the training data to achieve better accuracy?

1. Modify the pipeline to track the system metrics to MLflow during the training step. For more information about gathering and logging systems metrics, check out [this link](https://mlflow.org/docs/latest/system-metrics/index.html).

1. Create a custom Metaflow card for the evaluation step to display a confusion matrix with the results of evaluating the model.

1. The evaluation step computes the model's accuracy and loss. Modify the code to calculate the model's average precision and recall and log them as metrics to MLflow.

1. The pipeline only registers the model when its accuracy exceeds a predefined threshold. Modify the code so the pipeline only registers the model when its accuracy is above the accuracy of the previous model registered in the MLflow model registry.

1. Modify the pipeline using a simple train-test split instead of cross-validation. The pipeline should split the dataset using stratified sampling into two separate sets, train the model using one set, and test it using the other.

1. The pipeline runs cross-validation and trains a final model using the entire dataset. Instead of building this final model, modify the code so the pipeline registers a custom model that uses the five models produced as part of the cross-validation process in an ensemble during inference time.

1. Metaflow supports resuming a pipeline from its last failed step. Modify the pipeline so it fails during the training process, then restart it from that point using Metaflow's resume capabilities.

1. Extend the pipeline to support multiple model types (e.g., logistic regression, random forest, XGBoost). Use a pipeline parameter to choose the model architecture, and log a comparison of performance metrics across models.