# The Inference Pipeline

We'll use a simple inference pipeline to make predictions using the model. 

To build this pipeline, we'll use a [custom MLflow model](https://mlflow.org/blog/custom-pyfunc) that will allow us to validate and transform the input requests, make predictions, and process the output from the model before returning a response to the client.

Here is what the inference pipeline looks like:

![Inference pipeline](.guide/inference-pipeline/images/inference.png)

The inference pipeline can access the SciKit-Learn transformation pipelines and the Keras model we fitted during training. It will use these artifacts to process the input request, generate a prediction, and prepare the response.

The pipeline will optionally store every input request and prediction. This step is critical if we want to monitor the model's performance.

You can run the tests associated with the inference pipeline by executing the following command:

```shell
uv run -- pytest -k test_model
```
