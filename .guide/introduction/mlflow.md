# Setting Up MLflow

[MLflow](https://mlflow.org/) is a platform-agnostic machine learning lifecycle management tool that will help us track experiments and share and deploy models. 

The different workflows we'll run as part of the project will connect to an MLflow server to store all of the data they generate and track and version the models. 

To run an MLflow server locally, open a terminal window and run the following command:

```shell
just mlflow
```

This recipe will start an MLflow server running on `127.0.0.1` and listening on port `5000`. We'll need to keep this server running while we work on the project.

Once running, you can navigate to [`http://127.0.0.1:5000`](http://127.0.0.1:5000) in a web browser to open MLflow's user interface. We'll use this interface to browse the experiments, models, and any other data we generate throughout the project.

For more information on how to run the MLflow server, check [Common ways to set up MLflow](https://mlflow.org/docs/latest/tracking.html#common-setups). You can also run the following command to get more information:

```shell
mlflow server --help
```
