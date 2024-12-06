# Setting Up MLflow

[MLflow](https://mlflow.org/) is a platform-agnostic machine learning lifecycle management tool that will help you track experiments and share and deploy models.

To run an MLflow server locally, open a terminal window and run the following command within the virtual environment we created earlier:

```shell
mlflow server --host 127.0.0.1 --port 5000
```

Once running, you can navigate to [`http://127.0.0.1:5000`](http://127.0.0.1:5000) in your web browser to open MLflow's user interface.

By default, MLflow tracks experiments and stores data in files inside a local `./mlruns` directory. You can change the location of the tracking directory or use a SQLite database using the parameter `--backend-store-uri`. For more information, check some of the [common ways to set up MLflow](https://mlflow.org/docs/latest/tracking.html#common-setups). You can also run the following command to get more information about the server:

```shell
mlflow server --help
```
