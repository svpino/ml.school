# Cross-Validation Strategy

We want to use a cross-validation process to evaluate the model's performance and then train a final model using the entire dataset. 

We'll use 5-fold cross-validation to train and evaluate five separate models and average their accuracy to determine the final evaluation score of the system.

![Cross-validation](.guide/training-pipeline/images/cross-validation.png)

By splitting the dataset into multiple folds and training and evaluating on different splits, we reduce the risk of overestimating or underestimating performance compared to using a single train-test split.

In the `cross_validation` step, we use a Metaflow [`foreach`](.guide/introduction-to-metaflow/foreach.md) to create an independent branch for each one of the folds. Each branch will transform the data, train a model, and evaluate it. 

Finally, we'll join these separate branches and compute the average score using the individual evaluation of each model.

