# Generating Fake Labels

To monitor the model's performance, we need to label the data captured by the model to compare it against its predictions.

After [generating fake traffic](.guide/monitoring-pipeline/generating-fake-traffic.md), run the following command to automatically generate ground truth labels for all the data captured by the model:

```shell
just labels
```

This command runs the [Traffic pipeline](src/pipelines/traffic.py) with the `--mode labels` parameter. An alternative way to generate the labels is to run the pipeline directly:

```shell
uv run src/pipelines/traffic.py run --mode labels
```

The pipeline will connect to the backend's database, load any unlabeled data, and generate a fake label to serve as the ground truth for that particular sample. This pipeline is only helpful for testing the [Monitoring pipeline](src/pipelines/monitoring.py). In a production environment, you must determine the actual ground truth for every sample.

By default, the pipeline uses the `backend.Local` implementation to load the production data from a SQLite database. You can change the [backend implementation](src/inference/backend.py) by specifying the `--backend` property and use the `--config` parameter to supply a JSON configuration file to the pipeline:

```shell
uv run pipelines/traffic.py run \
    --mode labels \
    --config config config/local.json \
    --backend backend.Local
```

The pipeline relies on the `--ground-truth-quality` parameter to determine how close the fake ground truth information should be to the predictions the model generated. Setting this parameter to a value less than `1.0` will introduce noise to simulate inaccurate model predictions. By default, this parameter has a value of `0.8`. You can use the following command to set the parameter to a value of `0.5` to simulate predictions that are correct only half the time:

```shell
uv run src/pipelines/traffic.py run \
    --mode labels \
    --ground-truth-quality 0.5
```
