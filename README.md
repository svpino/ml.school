# Machine Learning School

This repository contains the source code of the [Machine Learning School](https://www.ml.school) program. Fork it to follow along.

If you find any problems with the code or have any ideas on improving it, please open an issue and share your recommendations.

## Configuration Setup

Here is a summary of the steps you need to follow:

1. Fork the program's [GitHub Repository](https://github.com/svpino/ml.school) and clone your fork on your local computer.

2. Go inside the folder of the program, create a virtual environment, and install the libraries from the `requirements.txt` file:

   ```bash
   $ python3 -m venv .venv
   $ source .venv/bin/activate
   $ python -m pip install --upgrade pip
   $ pip install -r requirements.txt
   ```

3. Install a Jupyter kernel on your virtual environment:

   ```bash
   $ pip install ipykernel
   $ python -m ipykernel install --user --name=.venv
   ```
   
4. If you don't have it, install [Docker](https://docs.docker.com/) on your computer. You can check if Docker is already installed and running using the following command:

   ```bash
   $ docker ps
   ```

5. [Install the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) and [configure it with your credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html).


6. Create an `.env` file in the root folder of the repository with the following content. Make sure you replace the value of each variable with the correct value:

   ```bash
    # Update this variable to your bucket name. This name must be unique 
    # across all AWS accounts.
    BUCKET=mlschool

    # These are your SageMaker Domain Id and User Profile. You can find
    # these values in your Amazon SageMaker dashboard under "Domains"
    DOMAIN_ID=d-givocgtibv1g
    USER_PROFILE=default-1682182522641

    # Update this variable to the ARN of the Execution Role associated to 
    # the user profile attached to your SageMaker Studio Domain.
    ROLE=arn:aws:iam::325223348818:role/service-role/AmazonSageMaker-ExecutionRole-20230312T160501
   ```
7. Open the Amazon IAM service, find the Execution Role you used in the previous step and edit the custom Execution Policy assigned to it. Edit the permissions of the Execution Policy and replace them with the contents of the `execution-policy.json` file. These permissions will give the Execution Role access to the resources we'll use during the program.

8. Now find the "Trust relationships" section under the same Execution Role, edit the configuration, and replace it with the contents of the `trust-relationships.json` file.

9. If your local environment is running on a Mac with an M-series chip, you need to build a TensorFlow docker image to train and evaluate the model we'll build in the program. You can do so with the following command:

   ```bash
   $ docker build -t sagemaker-tensorflow-training-toolkit-local penguins/container/.
   ```

10. If you are planning to run the notebook from inside SageMaker Studio, you need to create a Lifecycle Configuration to update the kernel. Load and execute the [studio-setup.ipynb](penguins/studio-setup.ipynb) notebook once inside SageMaker Studio. After doing this, you can open the [cohort.ipynb](penguins/cohort.ipynb) notebook and use the "TensorFlow 2.11 Python 3.9 CPU Optimized" kernel with the start-up script named "ml.school."


## Resources

* [Serving a TensorFlow model from a Flask application](penguins/serving/flask/README.md): A simple Flask application that serves a multi-class classification TensorFlow model to determine the species of a penguin.


## Assignments

Here is a list of assignments you can complete to practice the concepts we cover during the program.

### Session 1

* **Assignment 1.1**: The SageMaker Pipeline we built supports running a few steps in Local Mode. The goal of this assignment is to run the pipeline on your local environment using Local Mode.
* **Assignment 1.2**: For this assignment, we want to run the end-to-end pipeline in SageMaker Studio. Ensure you turn off Local Mode before doing so.
* **Assignment 1.3**: The pipeline uses Random Sampling to split the dataset. Modify the code to use Stratified Sampling instead.
* **Assignment 1.4**: For this assignment, we want to run a distributed Processing Job across multiple instances to capitalize the `island` column of the dataset. Your dataset will consist of 10 different files stored in S3. Set up a Processing Job using two instances. When specifying the input to the Processing Job, you must set the `ProcessingInput.s3_data_distribution_type` attribute to `ShardedByS3Key`. By doing this, SageMaker will run a cluster with two instances simultaneously, each with access to half the files.
* **Assignment 1.5**: Pipeline steps can encounter exceptions. In some cases, retrying can resolve these issues. For this assignment, configure the Processing Step so it automatically retries the step a maximum of 5 times if it encounters an `InternalServerError`. Check the [Retry Policy for Pipeline Steps](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-retry-policy.html) documentation for more information.

### Session 2

* **Assignment 2.1**: The training script trains the model using a hard-coded learning rate value. Modify the code to accept the learning rate as a parameter we can control from outside the script.
* **Assignment 2.2**: We currently define the number of epochs to train the model as a constant that we pass to the Estimator using the list of hyperparameters. Replace this constant with a new Pipeline Parameter named `training_epochs`. You'll need to specify this new parameter when creating the Pipeline.
* **Assignment 2.3**: The current tuning process aims to find the model with the highest validation accuracy. Modify the code to focus on the model with the lowest training loss.
* **Assignment 2.4**: We used an instance of [`SKLearnProcessor`](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html#scikit-learn-processor) to run the script that transforms and splits the data, but there's no way to add additional dependencies to the processing container. Modify the code to use an instance of [`FrameworkProcessor`](https://sagemaker.readthedocs.io/en/stable/api/training/processing.html#sagemaker.processing.FrameworkProcessor) instead. This class will allow you to specify a directory containing a `requirements.txt` file containing a list of dependencies. SageMaker will install these libraries in the processing container before triggering the processing job.
* **Assignment 2.5:** We want to execute the pipeline whenever the dataset changes. We can accomplish this by using [Amazon EventBridge](https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-what-is.html). Configure an event to automatically start the pipeline when a new file is added to the S3 bucket where we store our dataset. Check [Amazon EventBridge Integration](https://docs.aws.amazon.com/sagemaker/latest/dg/pipeline-eventbridge.html) for an implementation tutorial.


### Session 3

* **Assignment 3.1**: The evaluation script computes the accuracy of the model and exports it as part of the evaluation report. Extend the evaluation report by adding the precision and the recall of the model on each one of the classes.
* **Assignment 3.2**: The Condition Step uses a hard-coded threshold value to determine if the model's accuracy is good enough to proceed. Modify the code so the pipeline uses the accuracy of the latest registered model version as the threshold. We want to register a new model version only if its performance is better than the previous version we registered.
* **Assignment 3.3**: We are using a QualityCheck Step to compute the baseline statistics and constraints from the input dataset. Right now, this step runs every time the pipeline runs. Modify the code to prevent the pipeline from continuing runing if the QualityCheck Step detects any violation on the data. Check [the different configuration settings of the QualityCheck Step](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-quality-clarify-baseline-lifecycle.html) for more information on how to accomplish this.
* **Assignment 3.4**: The current pipeline uses either a Training Step or a Tuning Step to build a model. Modify the pipeline to use both steps at the same time. The evaluation script should evaluate the model coming from the Training Step and the best model coming from the Tuning Step and output the accuracy and location in S3 of the best model. You should modify the code to register the model assets specified in the evaluation report.
* **Assignment 3.5**: Instead of running the entire pipeline from start to finish, sometimes you may only need to iterate over particular steps. SageMaker Pipelines supports [Selective Execution for Pipeline Steps](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-selective-ex.html). In this assignment you will use Selective Execution to only run one specific step of the pipeline. [Unlocking efficiency: Harnessing the power of Selective Execution in Amazon SageMaker Pipelines](https://aws.amazon.com/blogs/machine-learning/unlocking-efficiency-harnessing-the-power-of-selective-execution-in-amazon-sagemaker-pipelines/) is a great article that explains this feature.
