# Loading the Data

After having a running pipeline, the next step is to load the data we'll use to train the model.

Since we are working with a simple data file, we can include it when running the Metaflow pipeline. We can then convert the raw text file into a Pandas data frame and make it available to the pipeline as a Metaflow artifact.

***Note:** Including the dataset when running the pipeline works well when working locally with a simple data file. For a production application, consider supplying the location of the data and loading it as part of the pipeline.*

To avoid duplicating any code, we can create a `DatasetMixin` base class in the [`common.py`](pipelines/common.py) file and move the dataset loading functionality into it. Every pipeline that requires loading the data can inherit from this base class to access the functionality. For example, notice how the [Training](pipelines/training.py) and the [Monitoring](pipelines/monitoring.py) pipelines inherit from the `DatasetMixin` class.

Finally, we want to clean the `sex` column from the original dataset to remove any extraneous values and shuffle the data before returning it. We discovered the issue with the `sex` column during the [Exploratory Data Analysis](notebooks/eda.ipynb) process.

Metaflow supports a `current.is_production` property to indicate whether the pipeline is running in development or production mode. By default, the property returns `False`. To execute the pipeline in production mode, run it with the `--production` attribute:

```shell
uv run -- python pipelines/training.py \ 
    --with retry \
    --environment conda --production run
```

You can run the [tests](tests/test_common_flowmixin.py) associated with the `FlowMixin` class by executing the following command:

```shell
uv run -- pytest -k test_common_flowmixin
```