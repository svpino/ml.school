# Generating Fake Labels

To monitor the model's performance, we need to label the data captured by the model to compare it against its predictions.

After generating fake traffic, you can use the Labels pipeline to automatically generate ground truth data for all the data captured by the model:

```shell
uv run -- python pipelines/labels.py \
    --environment conda run
```

You can also use the `just` command with the `labels` recipe:

```shell
just labels
```

The Labels pipeline will connect to the model backend, load any unlabeled data, and generate a fake label to serve as the ground truth for that particular sample. This pipeline is only helpful for testing the Monitoring pipeline. In a production environment, you must determine the actual ground truth for every sample.

By default, the pipeline uses the `backend.Local` implementation to load the production data from a SQLite database. You can change the [backend implementation](pipelines/inference/backend.py) by specifying the `--backend` property:

```shell
uv run -- python pipelines/labels.py \
    --environment conda run \
    --backend backend.Local
```

To provide configuration settings to a specific backend implementation, you can use the `--config` parameter to supply a JSON configuration file to the pipeline. The [`config/local.json`](config/local.json) file is an example configuration file for the [`backend.Local`](pipelines/inference/backend.py) backend. You can use this file as follows:

```shell
uv run -- python pipelines/labels.py \
    --config config config/local.json \
    --environment conda run \
    --backend backend.Local
```

The Labels pipeline relies on the `--ground-truth-quality` parameter to determine how close the fake ground truth information should be to the predictions the model generated. Setting this parameter to a value less than `1.0` will introduce noise to simulate inaccurate model predictions. By default, this parameter has a value of `0.8`.
