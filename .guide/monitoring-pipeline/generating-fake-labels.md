# Generating Fake Labels

To monitor the model's performance, we need to label the data captured by the model to compare it against its predictions.

After [generating fake traffic](.guide/monitoring-pipeline/generating-fake-traffic.md), you can use the Labels pipeline to automatically generate ground truth data for all the data captured by the model:

```shell
just labels
```
If you don't want to use the recipe, you can execute the following command:

```shell
uv run pipelines/labels.py run
```

The Labels pipeline will connect to the backend's database, load any unlabeled data, and generate a fake label to serve as the ground truth for that particular sample. This pipeline is only helpful for testing the Monitoring pipeline. In a production environment, you must determine the actual ground truth for every sample.

By default, the pipeline uses the `backend.Local` implementation to load the production data from a SQLite database. You can change the [backend implementation](pipelines/inference/backend.py) by specifying the `--backend` property and use the `--config` parameter to supply a JSON configuration file to the pipeline:

```shell
uv run pipelines/labels.py run \
    --config config config/local.json \
    --backend backend.Local
```

The Labels pipeline relies on the `--ground-truth-quality` parameter to determine how close the fake ground truth information should be to the predictions the model generated. Setting this parameter to a value less than `1.0` will introduce noise to simulate inaccurate model predictions. By default, this parameter has a value of `0.8`. You can use the following command to set the parameter to a value of `0.5` to simulate predictions that are correct only half the time:

```shell
uv run pipelines/labels.py run \
    --ground-truth-quality 0.5
```
