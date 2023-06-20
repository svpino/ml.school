# Machine Learning School

This project is part of the [Machine Learning School](https://www.ml.school) program.

* The [Penguins in Production](penguins-cohort.ipynb) notebook: An Amazon SageMaker pipeline hosting a multi-class classification model for the [Penguins dataset](https://www.kaggle.com/parulpandey/palmer-archipelago-antarctica-penguin-data).
* The [Pipeline of Digits](mnist/mnist.ipynb) notebook: A starting notebook for solving the "Pipeline of Digits" assignment.

## How should you approach the assignments?

Before looking at the assignments, focus on running the Penguins pipeline from top to bottom. 90% of the value this cohort offers you will come from understanding and executing the Penguins pipeline end-to-end.

Please, spend some time after every session, and make sure everything we discussed during the session is clear.

When you finish the session, look at the assignments and choose the ones you think will help you the most. For instance, I have some tasks related to using PyTorch instead of TensorFlow. If you don't need PyTorch, feel free to ignore those assignments.

Again, first and foremost, focus on the Penguins pipeline, and only then look at the assignments.

## Session 1 - Data Preprocessing

### Assignments

1. If you can't access an existing AWS Account, set up a new account. Create a user that belongs to the "administrators" User Group. Ensure you use MFA (Multi-Factor Authentication).

2. Set up an Amazon SageMaker domain and launch SageMaker Studio.

3. Create a GitHub repository and clone it from inside SageMaker Studio. You'll use this repository to store the code used during this program.

4. Configure your SageMaker Studio session to store your name and email address and cache your credentials. You can use the following commands from a Terminal window:

```bash
$ git config --global user.name "John Doe"
$ git config --global user.email johndoe@example.com
$ git config --global credential.helper store
```

5. Throughout the course, you will work on the "Pipeline of Digits" project to set up a SageMaker pipeline for a simple computer vision project. For this assignment, open the [`mnist.ipynb`](mnist/mnist.ipynb) notebook and follow the instructions to prepare everything you need to start the project.

6. Set up a SageMaker pipeline for the "Pipeline of Digits" project. Create a Processing Step where you split 20% off the MNIST train set to use as a validation set.


## Session 2 - Model Training and Tuning

### Assignments

1. Modify the training script to accept the `learning_rate` as a new hyperparameter.

2. If you prefer PyTorch, replace the TensorFlow Estimator with a PyTorch Estimator. Check the [Use PyTorch with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#create-an-estimator) page for an example of how to create a PyTorch Estimator.

3. Modify the Hyperparameter Tuning Job to find the best `learning_rate` value between `0.01` and `0.03`. Check the [ContinuousParameter](https://sagemaker.readthedocs.io/en/stable/api/training/parameter.html#sagemaker.parameter.ContinuousParameter) class for more information on how to configure this parameter.

4. Modify the SageMaker Pipeline you created for the "Pipeline of Digits" project and add a Training Step. This Training Step should receive the train and validation splits.


## Session 3 - Model Registration

### Assignments

1. The evaluation script produces an evaluation report containing the accuracy of the model. Extend the evaluation report by adding other metrics. For example, add the support of the test set (the number of samples in the test set.)

2. Modify your pipeline to add a new [Condition Step](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-condition) that's called if the model's accuracy is not above the specified threshold. Set the condition to succeed if the accuracy is above 50% and register the model as "PendingManualApproval." Don't register the model if the accuracy is not greater or equal to 50%. In summary, register the model as "Approved" if its accuracy is greater or equal to 70% and as "PendingManualApproval" if its accuracy is greater or equal to 50%.

3. If you run the Training and Tuning Steps simultaneously, create two different Evaluation Steps to evaluate both models independently.

4. Instead of running the Training and Tuning Steps simultaneously, run the Tuning Step but create two evaluation steps to evaluate the two best models produced by the Tuning Step. Check the [TuningStep.get_top_model_s3_uri()](https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html#sagemaker.workflow.steps.TuningStep.get_top_model_s3_uri) function to retrieve the two best models.

5. Modify the SageMaker Pipeline you created for the "Pipeline of Digits" project and add an evaluation and a registration step.


## Session 4 - Model Deployment

### Assignments

1. The custom inference code we built doesn't support processing more than one sample simultaneously. Modify the inference script to allow processing multiple samples at the same time. The output should be an array of JSON objects containing the prediction and the confidence corresponding to each input sample.

2. Load the test data and run every sample through the endpoint using a Predictor. Build a simple function that computes the accuracy on this test set.

3. Customize the inference process of the "Pipeline of Digits" project endpoint to receive a JSON containing an image URL and return the digit in the image.

4. Modify the SageMaker Pipeline you created for the "Pipeline of Digits" project and add a Lambda Step to deploy the model automatically.


## Session 5 - Data Monitoring

### Assignments

1. Modify the SageMaker Pipeline you created for the "Pipeline of Digits" project and add the necessary steps to generate a Data Quality baseline.

2. Build a simple function that generates fake traffic to the "Pipeline of Digits" endpoint so we can start monitoring the quality of the data coming in.

3. Modify the SageMaker Pipeline you created for the "Pipeline of Digits" project and add a new [Lambda Step](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-lambda) to schedule the Data Quality Monitoring Job automatically.


## Session 6 - Model Monitoring

### Assignments

1. Modify the SageMaker Pipeline you created for the "Pipeline of Digits" project and add the necessary steps to generate a Model Quality baseline.

2. Build a simple function that generates fake ground truth data for the data captured by the "Pipeline of Digits" endpoint.

3. Modify the SageMaker Pipeline you created for the "Pipeline of Digits" project and add a new [Lambda Step](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-lambda) to schedule the Model Quality Monitoring Job automatically.

