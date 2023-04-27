# Machine Learning School

This project is part of the [Machine Learning School](https://www.ml.school) program.

* The [Penguins in Production](penguins.ipynb) notebook: An Amazon SageMaker pipeline hosting a multi-class classification model for the [Penguins dataset](https://www.kaggle.com/parulpandey/palmer-archipelago-antarctica-penguin-data).
* The [Pipeline of Digits](mnist/mnist.ipynb) notebook: A starting notebook for solving the "Pipeline of Digits" assignment.

## Session 1 - Getting Started

This session aims to build a simple [SageMaker Pipeline](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html) with one step to preprocess the [Penguins dataset](https://www.kaggle.com/parulpandey/palmer-archipelago-antarctica-penguin-data). We'll use a [Processing Step](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-processing) with a [SKLearnProcessor](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html#scikit-learn-processor) to execute a preprocessing script. Check the [SageMaker Pipelines Overview](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html) for an introduction to the fundamental components of a SageMaker Pipeline.

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

This session extends the [SageMaker Pipeline](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html) we built in the previous session with a step to train a model. We'll explore the [Training](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-training) and the [Tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-tuning) steps. 

### Assignments

1. Modify the training script to accept the `learning_rate` as a new hyperparameter. You can use the list of hyperparameters supplied to the Estimator.

2. Replace the TensorFlow Estimator with a PyTorch Estimator. Check the [Use PyTorch with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#create-an-estimator) page for an example of how to create a PyTorch Estimator. You'll need to create a new training script that builds a PyTorch model to solve the problem.

3. Modify the Hyperparameter Tuning Job to find the best `learning_rate` value between `0.01` and `0.03`. Check the [ContinuousParameter](https://sagemaker.readthedocs.io/en/stable/api/training/parameter.html#sagemaker.parameter.ContinuousParameter) class for more information on how to configure this parameter.

4. Modify the SageMaker Pipeline to run the Training and Tuning Step concurrently. This is not something you'd do in an actual application, but it's an excellent exercise to understand how the different steps coexist in the same pipeline.

5. Modify the SageMaker Pipeline you created for the "Pipeline of Digits" project and add a Training Step. This Training Step should receive the training and validation data from the Processing Step you created in Session 1.


## Session 3 - Model Evaluation

This session extends the [SageMaker Pipeline](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html) with a step to evaluate the model. We'll use a [Processing Step](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-processing) with a [ScriptProcessor](https://sagemaker.readthedocs.io/en/stable/api/training/processing.html#sagemaker.processing.ScriptProcessor) running TensorFlow to execute an evaluation script. 

### Assignments

1. The evaluation script produces an evaluation report containing the accuracy of the model. Extend the evaluation report by adding other metrics. For example, add the support of the test set (the number of samples in the test set.)

2. One of the assignments from the previous Session was to replace the TensorFlow Estimator with a PyTorch Estimator. You can now modify the evaluation step to load a script that uses PyTorch to evaluate the model.

3. If you run the Training and Tuning Steps simultaneously, create two different Evaluation Steps to evaluate both models independently.

4. Instead of running the Training and Tuning Steps simultaneously, run the Tuning Step but create two evaluation steps to evaluate the two best models produced by the Tuning Step. Check the [TuningStep.get_top_model_s3_uri()](https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html#sagemaker.workflow.steps.TuningStep.get_top_model_s3_uri) function to retrieve the two best models.

5. Modify the SageMaker Pipeline you created for the "Pipeline of Digits" project and add an evaluation step that receives the test data from the preprocessing step.


## Session 4 - Model Registration

This session extends the [SageMaker Pipeline](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html) with a step to register a new model if it reaches a predefined accuracy threshold. We'll use a [Condition Step](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-condition) to determine whether the model's accuracy is above a threshold and a [Model Step](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-model) to register the model. After we register the model, we'll deploy it manually. To learn more about the Model Registry, check [Register and Deploy Models with Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html).

### Assignments

1. Modify your pipeline to add a new [Lambda Step](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-lambda) that's only called if the model's accuracy is not above the specified threshold. What you decide to do in the Lambda function is optional.

2. Modify your pipeline to add a new [Condition Step](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-condition) that's called if the model's accuracy is not above the specified threshold. Set the condition to succeed if the accuracy is above 50% and register the model as "PendingManualApproval." Don't register the model if the accuracy is not greater or equal to 50%. In summary, register the model as "Approved" if its accuracy is greater or equal to 70% and as "PendingManualApproval" if its accuracy is greater or equal to 50%.

3. Modify the payload you send to the endpoint so you can classify multiple examples simultaneously. 

4. Modify the SageMaker Pipeline you created for the "Pipeline of Digits" project and add a step to register the model.


## Session 5 - Model Deployment

This session extends the [SageMaker Pipeline](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html) with a step to deploy the model to an endpoint automatically. We'll use a [Lambda Step](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-lambda) to create an endpoint and deploy the model. To control the endpoint's inputs and outputs, we'll modify the model's assets to include code that customizes the processing of a request. 

### Assignments

1. The custom inference code we built during the Session accepts JSON requests. Modify the code to receive the input data in CSV or JSON format.

2. Load the test data and run every sample through the endpoint using a Predictor. Check the data the endpoint captured by downloading the files from the S3 location where you stored them.

3. Customize the inference process of the "Pipeline of Digits" project endpoint to receive a JSON containing an image URL and return the digit in the image.

4. Modify the SageMaker Pipeline you created for the "Pipeline of Digits" project and add a Lambda Step to deploy the model automatically.


## Session 6 - Model Monitoring

This session aims to set up a monitoring process to analyze the quality of the data and the model. For this, we will have SageMaker capture and evaluate the data observed by the endpoint.

### Assignments

1. Modify the SageMaker Pipeline you created for the "Pipeline of Digits" project and add the necessary steps to generate a Model Quality baseline. Since the endpoint expects an image URL, we don't need to worry about data quality.

2. Schedule a Model Quality Monitoring Job to monitor the "Pipeline of Digits" model. Generate fake ground truth data like we did during Session 6.

3. Modify the SageMaker Pipeline you created for the "Pipeline of Digits" project and add a new [Lambda Step](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-lambda) to schedule the Model Quality Monitoring Job automatically.

