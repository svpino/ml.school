# The Training Pipeline

The Training pipeline trains, evaluates and registers a model that predicts the species of a penguin given its information. Every time it runs, the pipeline registers a new version of the model in the [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html).

![Training pipeline](.guide/training-pipeline/images/training.png)

To run the [Training pipeline](src/pipelines/training.py) locally, make sure that MLflow is running and execute the command below (if MLflow is not running, the pipeline will fail with a message "Failed to connect to MLflow server"):

```shell
just train
```

The Training pipeline loads and transforms the `penguins.csv` dataset, trains a model, evaluates its performance, and registers the model in the model registry. After running the pipeline, you should see a new version of the `penguins` model in the model registry.

The pipeline registers the model only if its accuracy is above a predefined threshold. You can change this threshold by specifying the `accuracy-threshold` parameter when running the pipeline:

```shell
uv run src/pipelines/training.py --with retry run \
    --accuracy-threshold 0.9
```

To display the supported parameters of the Training pipeline, run the following command:

```shell
uv run src/pipelines/training.py run --help
```

You can observe the execution of the pipeline and visualize its results live by running a Metaflow card server using the following command:

```shell
uv run src/pipelines/training.py card server
```

After the card server is running, open your browser and navigate to [localhost:8324](http://localhost:8324/). Every time you run the Training pipeline, the viewer will automatically update to show the cards related to the latest pipeline execution.