# Machine Learning School

This repository contains the source code of the [Machine Learning School](https://www.ml.school) program. Fork it to follow along.

If you find any problems with the code or have any ideas on improving it, please, open and issue and share your recommendations.

## Penguins

During this program, we'll create a [SageMaker Pipeline](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html) to build an end-to-end Machine Learning system to solve the problem of classifying penguin species.

Here are the relevant notebooks:

* The [Setup notebook](penguins/penguins-setup.ipynb): We'll use this notebook at the beginning of the program to setup SageMaker Studio. You only need to go through the code here once.
* The [Penguins in Production notebook](penguins/penguins-cohort.ipynb): This is the main notebook we'll use during the program. Inside you'll find the code of every session.

## Questions

Answering these questions will help you understand the material discussed during this session. Notice that each question could have one or more correct answers.

```html
   <div style="margin: 30px 0 10px 0;"><span style="font-size: 1.1em; padding:4px; background-color: #b8bf9f; color: #000;"><strong>Question 1.1</strong></span></div>
```

What will happen if we apply the SciKit-Learn transformation pipeline to the entire dataset before splitting it?

1. Scaling will use the global statistics of the dataset, leaking the mean and variance of the test samples into the training process.
2. Imputing the missing numeric values will use the global mean, leading to data leakage.
3. The transformation pipeline expects multiple sets so it wouldn't work.
4. We will reduce the number of lines of code we need to transform the dataset.


<div style="margin: 30px 0 10px 0;"><span style="font-size: 1.1em; padding:4px; background-color: #b8bf9f; color: #000;"><strong>Question 1.2</strong></span></div>

A hospital wants to predict which patients are prone to get a disease based on their medical history. They use weak supervision to label the data using a set of heuristics automatically. What are some of the disadvantages of weak supervision?

1. Weak supervision doesn't scale to large datasets.
2. Weak supervision doesn't adapt well to changes requiring relabeling.
3. Weak supervision produces noisy labels.
4. We might be unable to use weak supervision to label every data sample.


<div style="margin: 30px 0 10px 0;"><span style="font-size: 1.1em; padding:4px; background-color: #b8bf9f; color: #000;"><strong>Question 1.3</strong></span></div>

When collecting the information about the penguins, the scientists encountered a few rare species. To prevent these samples from not showing when splitting the data, they recommended using Stratified Sampling. Which of the following statements about Stratified Sampling are correct?

1. Stratified Sampling assigns every sample of the population an equal chance of being selected.
2. Stratified Sampling preserves the original distribution of different groups in the data.
3. Stratified Sampling requires having a larger dataset compared to Random Sampling.
4. Stratified Sampling can't be used when is not possible to divide all samples into groups.


<div style="margin: 30px 0 10px 0;"><span style="font-size: 1.1em; padding:4px; background-color: #b8bf9f; color: #000;"><strong>Question 1.4</strong></span></div>

Using more features to build a model will not necessarily lead to better predictions. Which of the following are drawbacks from adding more features?

1. More features in a dataset increases the opportunity for data leakage.
2. More features in a dataset increases the opportunity for overfitting.
3. More features in a dataset increases the memory necessary to serve a model.
4. More features in a dataset increases the development and maintenance time of a model. 


<div style="margin: 30px 0 10px 0;"><span style="font-size: 1.1em; padding:4px; background-color: #b8bf9f; color: #000;"><strong>Question 1.5</strong></span></div>

A bank wants to store every transaction they handle in a set of files in the cloud. Each file will contain the transactions generated in a day. The team who will manage these files wants to optimize the storage space and downloding speed. What format should the bank use to store the transactions?

1. The bank should store the data in JSON format.
2. The bank should store the data in CSV format.
3. The bank should store the data in Parquet format.
4. The bank should store the data in Pandas format.


## Pipeline of Digits

During the program, you are encouraged to work on the Pipeline of Digits problem as the main assignment. To make it easier to start, you can use the [Pipeline of Digits](mnist/mnist.ipynb) as a starting point.

## Resources

* [Serving a TensorFlow model from a Flask application](penguins/serving/flask/README.md): A simple Flask application that serves a multi-class classification TensorFlow model to determine the species of a penguin.

