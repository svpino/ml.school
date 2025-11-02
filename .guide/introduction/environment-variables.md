# Environment Variables

As we make progress through the program, we'll need to set up a few environment variables to run the project. To keep everything organized, we'll use a `.env` file to store these variables and load them into our Python code.

To create this file, you can run the following command:

```shell
just env
```

If the `.env` file doesn't exist, this recipe will create it with the following variables:

* `KERAS_BACKEND`: This variable specifies the backend [Keras](https://keras.io/) should use when building and running models.
* `MLFLOW_TRACKING_URI`: This variable points to the MLflow server's URI. We'll use it throughout different workflows to connect to the server.

After creating the `.env` file, the recipe will load the environment variables into your active shell. You can check that everything worked by printing the value of one of these variables:

```shell
echo $KERAS_BACKEND
```

This command should print `tensorflow` in the terminal, indicating the variables were correctly set.