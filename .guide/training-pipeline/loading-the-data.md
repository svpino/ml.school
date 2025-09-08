# Loading the Data

After setting up the basic building blocks of the pipeline, the next step is to load the data that we'll use to train the model.

To avoid duplicating any code, we'll create a [`pipeline.py`](src/common/pipeline.py) file and move the dataset loading functionality into it. Every pipeline that requires loading the data will inherit from the [`Pipeline`](src/common/pipeline.py) base class. For example, notice how the [`Training`](src/pipelines/training.py) and the [`Monitoring`](src/pipelines/monitoring.py) pipelines inherit from the [`Pipeline`](src/common/pipeline.py) class.

To load the dataset, we'll combine a [`Property`](.guide/introduction-to-metaflow/parameterizing-flows.md) and a [custom decorator](.guide/introduction-to-metaflow/decorators-and-mutators.md). We'll use the property `dataset` to specify the location of the data and the `@dataset` decorator to load the data at any point in the flow.

The `@dataset` decorator loads the dataset in memory, cleans the `sex` column to remove any extraneous values, and shuffles the data before returning it. We discovered the issue with the `sex` column during the [Exploratory Data Analysis](notebooks/eda.ipynb) process.

When you run the [`Training`](src/pipelines/training.py) pipeline, the flow will load the default [`penguins.csv`](data/penguins.csv) dataset from the `data` directory. You can specify a different file using the `--dataset` argument when running the pipeline:

```shell
uv run src/pipelines/training.py --with retry \
    --dataset data/other-dataset.csv
```

Metaflow supports a `current.is_production` property to indicate whether the pipeline is running in development or production mode. We can use this property to change the random seed we use to shuffle the dataset, so we never get the same shuffle if we are running in the production environment. To execute the pipeline in production mode, run it with the `--production` attribute:

```shell
uv run src/pipelines/training.py --with retry \
    --production run
```

Finally, to run the [tests](tests/common/test_common_pipeline.py) associated with the `Pipeline` base class, execute the following command:

```shell
uv run -- pytest -k test_common_pipeline
```