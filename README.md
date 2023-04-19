# Machine Learning School

This project is part of the [Machine Learning School](https://www.ml.school) program.

* The [Penguins in Production](penguins.ipynb) notebook: An Amazon SageMaker pipeline hosting a multi-class classification model for the [Penguins dataset](https://www.kaggle.com/parulpandey/palmer-archipelago-antarctica-penguin-data).
* The [Pipeline of Digits](mnist/mnist.ipynb) notebook: A starting notebook for solving the "Pipeline of Digits" assignment.

## Session 1

The goal of this session is to build a simple [SageMaker Pipeline](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html) with one step to preprocess the [Penguins dataset](https://www.kaggle.com/parulpandey/palmer-archipelago-antarctica-penguin-data). We'll use a [Processing Step](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-processing) with a [SKLearnProcessor](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html#scikit-learn-processor) to execute a preprocessing script.

### Assignments

1. If you can't access an existing AWS Account, set up a new account. Create a user that belongs to the "administrators" User Group. Ensure you use MFA (Multi-Factor Authentication).

2. Set up an Amazon SageMaker domain. Create a new execution role and ensure it has access to the S3 bucket you'll use during this class. You can also specify "Any S3 bucket" if you want this role to access every S3 bucket in your AWS account.

3. Create a GitHub repository and clone it from inside SageMaker Studio. You'll use this repository to store the code used during this program.

4. Configure your SageMaker Studio session to store your name and email address and cache your credentials. You can use the following commands from a Terminal window:

```bash
$ git config --global user.name "John Doe"
$ git config --global user.email johndoe@example.com
$ git config --global credential.helper store
```

5. Throughout the course, you will work on the "Pipeline of Digits" project with the goal of seting up a SageMaker pipeline for a simple computer vision project. For this assignment, open the [`mnist.ipynb`](mnist/mnist.ipynb) notebook and follow the instructions to prepare everything you need to start the project.

6. Setup a SageMaker pipeline for the "Pipeline of Digits" project. Create a Processing Step where you split 20% off the MNIST train set to use as a validation set.


## Session 2

This session extends the [SageMaker Pipeline](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html) we built in the previous session with a step to train a model. We'll explore the [Training](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-training) and the [Tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-tuning) steps.

### Assignments

1. Modify the training script so it accepts the `learning_rate` as a new hyperparameter. You can use the list of hyperparameters supplied to the Estimator to accomplish this.

2. Replace the TensorFlow Estimator with a PyTorch Estimator. Check the [Use PyTorch with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#create-an-estimator) page for an example of how to create a PyTorch Estimator. You'll need to create a new training script that builds a PyTorch model to solve the problem.

3. Modify the Hyperparameter Tuning Job to find the best `learning_rate` value between `0.01` and `0.03`. Check the [ContinuousParameter](https://sagemaker.readthedocs.io/en/stable/api/training/parameter.html#sagemaker.parameter.ContinuousParameter) class for more information on how to configure this parameter.

4. Modify the SageMaker Pipeline to run the Training Step and the Tuning Step concurrently. This is not something you'd do in a real application, but it's a good exercise to understand how the different steps can coexist in the same Pipeline.

5. Modify the SageMaker Pipeline you created for the "Pipeline of Digits" project and add a Training Step. This Training Step should receive the train and validation data from the Processing Step you created in Session 1.


## Session 3

This session extends the [SageMaker Pipeline](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html) with a step to evaluate the model. We'll use a [Processing Step](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-processing) with a [ScriptProcessor](https://sagemaker.readthedocs.io/en/stable/api/training/processing.html#sagemaker.processing.ScriptProcessor) running TensorFlow to execute an evaluation script. 

### Assignments

1. The evaluation script produces an evaluation report containing the accuracy of the model. Extend the evaluation report by adding other metrics. For example, add the support of the test set (the number of samples in the test set.)

2. One of the assignments from the previous session was to replace the TensorFlow Estimator with a PyTorch Estimator. You can now modify the evaluation step to load a script that uses PyTorch to evaluate the model.

3. If you are runing the Training and Tuning Steps simultaneously, create two different Evaluation Steps to evaluate both models independently.

4. Instead of runing the Training and Tuning Steps simultaneously, run the Tuning Step but create two evaluation steps to evaluate the two best models produced by the Tuning Step. Check the [TuningStep.get_top_model_s3_uri()](https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html#sagemaker.workflow.steps.TuningStep.get_top_model_s3_uri) function to retrieve the two best models.

5. Modify the SageMaker Pipeline you created for the "Pipeline of Digits" project and add an evaluation step that receives the test data from the preprocessing step.