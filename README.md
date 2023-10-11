# Machine Learning School

This repository contains the source code of the [Machine Learning School](https://www.ml.school) program. Fork it to follow along.

If you find any problems with the code or have any ideas on improving it, please open an issue and share your recommendations.

## Penguins

During this program, we'll create a [SageMaker Pipeline](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html) to build an end-to-end Machine Learning system to solve the problem of classifying penguin species.

Here are the relevant notebooks:

* The [Setup notebook](penguins/penguins-setup.ipynb): We'll use this notebook at the beginning of the program to set up SageMaker Studio. You only need to go through the code here once.
* The [Penguins in Production notebook](penguins/penguins-cohort.ipynb): This is the main notebook we'll use during the program. Inside, you'll find the code of every session.

## Pipeline of Digits

During the program, you are encouraged to work on the Pipeline of Digits problem as the main assignment. To make starting easier, you can use the [Pipeline of Digits](mnist/mnist.ipynb) as a starting point.

## Resources

* [Serving a TensorFlow model from a Flask application](penguins/serving/flask/README.md): A simple Flask application that serves a multi-class classification TensorFlow model to determine the species of a penguin.


## Multi-Choice Questions

Answering these questions will help you understand the material discussed during the program. Notice that each question could have one or more correct answers.



#### Question 3.1
Why do we use the `SparseCategoricalCrossentropy` loss function to train our model instead of the `CategoricalCrossentropy` function?

1. Because our target column contains integer values.
2. Because our target column is one-hot encoded.
3. Because our target column contains categorical values.
4. Because there are a lot of sparse values in our dataset.

#### Question 3.2
When a Training Job finishes, SageMaker automatically uploads the model to S3. Which of the following statements about this process is correct?

1. SageMaker automatically creates a `model.tar.gz` file with the entire content of the `/opt/ml/model` directory.
2. SageMaker automatically creates a `model.tar.gz` file with any files inside the `/opt/ml/model` directory as long as those files belong to the model we trained.
3. SageMaker automatically creates a `model.tar.gz` file with any new files created inside the container by the training script.
4. SageMaker automatically creates a `model.tar.gz` file with the output folder content configured in the training script.

#### Question 3.3
Our pipeline uses "file mode" to provide the Training Job access to the dataset. When using file mode, SageMaker downloads the training data from S3 to a local directory in the training container. Imagine we have a large dataset and don't want to wait for SageMaker to download every time we want to train a model. How can we solve this problem?

1. We can train our model with a smaller portion of the dataset.
2. We can increase the number of instances and train many models in parallel.
3. We can use "fast file mode" to get file system access to S3.
4. We can use "pipe mode" to stream data directly from S3 into the training container.

#### Question 3.4
Which of the following statements are true about the usage of `max_jobs` and `max_parallel_jobs` when running a Hyperparameter Tuning Job?

1. `max_jobs` represents the maximum number of Training Jobs the Hyperparameter Tuning Job will start. 
2. `max_parallel_jobs` represents the maximum number of Training Jobs that will run in parallel at any given time during a Hyperparameter Tuning Job.
3. `max_parallel_jobs` can never be larger than `max_jobs`.
4. `max_jobs` can never be larger than `max_parallel_jobs`.

#### Question 3.5
Which statements are true about tuning hyperparameters as part of a pipeline?

1. Hyperparameter Tuning Jobs that don't use Amazon algorithms require a regular expression to extract the objective metric from the logs.
2. When using a Tuning Step as part of a pipeline, SageMaker will create as many Hyperparameter Tuning Jobs as specified by the `HyperparameterTuner.max_jobs` attribute.
3. Hyperparameter Tuning Jobs support Bayesian, Grid Search, and Random Search strategies.
4. Using a Tuning Step is more expensive than a Training Step.

### Question 4.1
When registering a model in the Model Registry, we can specify a set of metrics stored with the model. Which of the following are some of the metrics supported by SageMaker?

1. Metrics that measure the bias in a model.
2. Metrics that help explain a model.
3. Metrics that measure the quality of the input data for a model.
4. Metrics that measure the quality of a model.

#### Question 4.2
We use the `Join` function to build the error message for the Fail Step. Imagine we want to build an Amazon S3 URI. What would be the output of executing `Join(on='/', values=['s3:/', "mlschool", "/", "12345"])`?

1. The output will be `s3://mlschool/12345`
2. The output will be `s3://mlschool/12345/`
3. The output will be `s3://mlschool//12345`
4. The output will be `s3:/mlschool//12345`

#### Question 4.3
Which of the following statements are correct about the Condition Step in SageMaker:

1. `ConditionComparison` is a supported condition type.
2. `ConditionIn` is a supported condition type.
3. When using multiple conditions together, the step will succeed if at least one of the conditions returns True.
4. When using multiple conditions together, they must return True for the step to succeed.

#### Question 4.4
Imagine we use a Tuning Step to run 100 Training Jobs. The best model should have the highest validation accuracy, but we mistakenly used "Minimize" as the objective type instead of "Maximize." The consequence is that the index of our best model is 100 instead of 0. How can we retrieve the best model from the Tuning Step?

1. We can use `TuningStep.get_top_model_s3_uri(top_k=0)` to retrieve the best model.
2. We can use `TuningStep.get_top_model_s3_uri(top_k=100)` to retrieve the best model.
3. We can use `TuningStep.get_bottom_model_s3_uri(top_k=0)` to retrieve the best model.
4. In this example, we can't retrieve the best model.

#### Question 4.5
If the model's accuracy is above the threshold, our pipeline registers it in the Model Registry. Which of the following functions are related to the Model Registry?

1. Model versioning: We can use the Model Registry to track different model versions, especially as they get updated or refined over time.
2. Model deployment: We can initiate the deployment of a model right from the Model Registry.
3. Model metrics: The Model Registry provides insights about a particular model through the registration of metrics.
4. Model features: The Model Registry lists every feature used to build the model.

#### Question 5.1
Imagine you created three models using the same Machine Learning framework. You want to host these models using SageMaker Endpoints. Multi-model endpoints provide a scalable and cost-effective solution for deploying many models in the same Endpoint. Which of the following statements are true regarding how Multi-model endpoints work?

1. SageMaker dynamically downloads the model from S3 and caches it in memory when you invoke the Endpoint. 
2. SageMaker automatically unloads unused models from memory when an Endpoint's memory utilization is high, and SageMaker needs to load another model.
3. You can dynamically add a new model to your Endpoint without writing any code. To add a model, upload it to the S3 bucket and invoke it through the Endpoint.
4. You can use the SageMaker SDK to delete a model from your Endpoint. 

#### Question 5.2
We enabled Data Capture in the Endpoint configuration. Data Capture is commonly used to record information that can be used for training, debugging, and monitoring. Which of the following statements are true about Data Capture?

1. SageMaker can capture the input traffic, the output responses, or both simultaneously.
2. If not specified, SageMaker captures 100% of requests.
3. The higher the traffic an Endpoint gets, the higher should be the sampling percentage used in the Data Capture configuration.
4. SageMaker supports Data Capture on always-running Endpoints but doesn't support it for Serverless Endpoints.

#### Question 5.3
Imagine you expect your model to have long idle periods. You don't want to run an Endpoint continuously; instead, you decide to use a Serverless Endpoint. Which of the following statements are true about Serverless Inference in SageMaker?

1. Serverless Inference scales the number of available endpoints to 0 when there are no requests.
2. Serverless Inference scales the number of available endpoints to 1 when there are no requests.
3. Serverless Inference solves the problem of cold starts when an Endpoint starts receiving traffic.
4. For Serverless Inference, you pay for the compute capacity used to process inference requests and the amount of data processed.

#### Question 5.4
Imagine you create an Endpoint with two different variants and assign each variant an initial weight of 1. How will SageMaker distribute the traffic between each variant?

1. SageMaker will send 100% of requests to both variants.
2. SageMaker will send 50% of requests to the first and 50% to the second variants.
3. SageMaker will send 100% of requests to the first variant and ignore the second one.
4. This scenario won't work because the sum of the initial weights across variants must be 1.

#### Question 5.5
Which attributes can you control when setting up auto-scaling for a model?

1. SageMaker will use the target metric to determine when and how much to scale.
2. The minimum capacity indicates the minimum number of instances that should be available.
3. The amount of time that should pass after a scale-in or a scale-out activity before another activity can start.
4. The algorithm that SageMaker will use to determine how to scale the model.

#### Question 6.1
To compute the data and the model quality baselines, we use the `train-baseline` and `test-baseline` outputs from the Preprocessing step of the pipeline. Which of the following is why we don't use the `train` and `test` outputs?

1. The `train` and `test` outputs are used in the Train and Evaluation steps, and SageMaker doesn't allow the reuse of outputs across a pipeline.
2. Computing the two baselines requires the data to be transformed with the SciKit-Learn pipeline we created as part of the Preprocessing step.
3. Computing the two baselines requires the data to be in its original format.
4. Computing the two baselines requires JSON data, but the `train` and `test` outputs are in CSV format.

#### Question 6.2
You build a computer vision model to recognize the brand and model of luxury handbags. After you deploy the model, one of the most important brands releases a new handbag that your model can't predict. How would you classify this type of model drift?

1. Sudden drift.
2. Gradual drift.
3. Incremental drift.
4. Reocurring drift.

#### Question 6.3
We use a custom script as part of the creation of the Data Monitoring schedule. Why do we need this custom script?

1. This script expands the input data with the fields coming from the endpoint output.
2. This script combines the input data with the endpoint output.
3. This script prevents the monitoring job from reporting superfluous violations.
4. This script expands the list of fields with the data SageMaker needs to detect violations.

#### Question 6.4
We created a function to generate labels for the data captured by the endpoint randomly. How does SageMaker know which label corresponds to a specific request?

1. SageMaker uses the timestamp of the request.
2. SageMaker uses the `inference_id` field we send to the endpoint on every request.
3. SageMaker uses the `event_id` field we send to the endpoint on every request.
4. SageMaker uses the `label_id` field we send to the endpoint on every request.

#### Question 6.5
Using our model, we use a Transform Step to generate predictions for the test data. When configuring this step, we filter the result from the step using the `output_filter` attribute. Assuming we configure this attribute with the value `$.SageMakerOutput['prediction','groundtruth']`, which of the following statements should be correct about the endpoint?

1. The endpoint should return a top-level field named `prediction`.
2. The endpoint should return a top-level field named `groundtruth`.
3. The endpoint should return a top-level field named `SageMakerOutput`.
4. The test dataset should include a field named `groundtruth`.
