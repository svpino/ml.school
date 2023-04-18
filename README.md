# Machine Learning School

This project is part of the [Machine Learning School](https://www.ml.school) program.

* The [Penguins in Production](penguins.ipynb) notebook: An Amazon SageMaker pipeline hosting a multi-class classification model for the [Penguins dataset](https://www.kaggle.com/parulpandey/palmer-archipelago-antarctica-penguin-data).
* The [Pipeline of Digits](mnist.ipynb) notebook: A starting notebook for solving the "Pipeline of Digits" assignment.

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

5. Throughout the course, you will work on the "Pipeline of Digits" project with the goal of seting up a SageMaker pipeline for a simple computer vision project. For this assignment, open the `mnist.ipynb` notebook and follow the instructions to prepare the dataset for the project.

6. Setup a SageMaker pipeline for the "Pipeline of Digits" project. Create a Processing Step where you split the MNIST dataset into a train and a test set.
