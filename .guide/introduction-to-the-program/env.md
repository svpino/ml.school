# Environment Variables

As we make progress through the program, we'll need to set up a few environment variables to run the project. To keep everything organized, we'll use a `.env` file to store these variables and load them into our Python code.

To create this file, you can run the following command:

```shell
just env
```

This recipe will create the `.env` file from scratch. Keep in mind that this command will overwrite any existing `.env` file in the project's root directory, so don't run it if you already have one and want to keep its content.

Here are the variables we'll set in the `.env` file:

* `KERAS_BACKEND`: This variable specifies the bakend [Keras](https://keras.io/) should use when building and running models.
* `MLFLOW_TRACKING_URI`: This variable points to the MLflow server's URI. We'll use it throughout different workflows to connect to the server.

After creating the `.env` file, load these variables into your active shell by run the following command:

```shell
export $(cat .env | xargs)
```