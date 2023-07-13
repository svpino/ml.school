# Machine Learning School

This repository contains the source code of the [Machine Learning School](https://www.ml.school) program. Fork it to follow along.

If you find any problems with the code or have any ideas on improving it, please, open and issue and share your recommendations.

## Penguins

During this program we'll create a [SageMaker Pipeline](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html) to build an end-to-end Machine Learning system to solve the problem of classifying penguin species.

Here are the relevant notebooks:

* The [Setup notebook](penguins/penguins-setup.ipynb): We'll use this notebook at the beginning of the program to setup SageMaker Studio. You only need to go through the code here once.
* The [Penguins in Production notebook](penguins/penguins-cohort.ipynb): This is the main notebook we'll use during the program. Inside you'll find the code of every session. 
* The [Endpoint notebook](penguins/penguins-endpoint.ipynb): This notebook contains routines and examples to interact with a SageMaker Endpoint.
* The [Monitoring notebook](penguins/penguins-monitoring.ipynb): This notebook contains the necessary code to configure and run data and model monitoring jobs.
* The [Teardown notebook](penguins/penguins-teardown.ipynb): You can use this notebook to remove some of the resources you created during the program.

## Pipeline of Digits

During the program, you are encouraged to work on the Pipeline of Digits problem as the main assignment. To make it easier to start, you can use the [Pipeline of Digits](mnist/mnist.ipynb) as a starting point.

