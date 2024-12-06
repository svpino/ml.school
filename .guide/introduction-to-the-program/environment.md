# Environment Variables

TBD

The final step is to create an `.env` file inside the project's root directory. We'll use this file to define the environment variables we'll need to run the pipelines:

```shell
echo "KERAS_BACKEND=jax" >> .env
```

Finally, we'll use the [`jq`](https://jqlang.github.io/jq/) command-line JSON processor to simplify some commands when working with different cloud environments, [`docker`](https://docs.docker.com/engine/install/) to deploy the model to the cloud, and [`just`](https://github.com/casey/just) to run project-specific commands. Make sure you have these tools installed in your system.


Modify the `.env` file inside the project's root directory and add the `MLFLOW_TRACKING_URI` environment variable. The following command will add the variable and export it in your current shell:

```shell
export $((echo "MLFLOW_TRACKING_URI=http://127.0.0.1:5000" >> .env; cat .env) | xargs)
```