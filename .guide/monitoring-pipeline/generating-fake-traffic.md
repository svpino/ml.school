# Generating Fake Traffic

Before running the Monitoring pipeline, we'll generate some fake traffic to the hosted model. We wouldn't need this for a production model that receives live traffic from users, but this will help us test the Monitoring pipeline.

To simplify the process, we can use the `Traffic` pipeline to send fake requests to the hosted model. This pipeline loads the original dataset, randomly selects a number of samples, and sends them to the hosted model in batches:

```shell
just traffic
```

If you don't want to use the recipe, you can execute the following command:

```shell
uv run pipelines/traffic.py run
```

Using the `--backend` parameter, you can specify how to communicate with the hosted model. This parameter expects the name of a class implementing the [`backend.Backend`](pipelines/inference/backend.py) abstract class. By default, this parameter will use the [`backend.Local`](pipelines/inference/backend.py) implementation, which knows how to submit requests to an inference server created using the `mlflow models serve` command.

You can use the `--config` parameter to supply a JSON configuration file to the pipeline. The [`config/local.json`](config/local.json) file is an example configuration file for the [`backend.Local`](pipelines/inference/backend.py) backend:

```shell
uv run pipelines/traffic.py run \
    --config config config/local.json \
    --backend backend.Local
```

By default, the Traffic pipeline will send 200 samples to the hosted model. If you want to send a different number, use the `--samples` parameter:

```shell
uv run pipelines/traffic.py run --samples 500
```

Finally, to evaluate whether the Monitoring pipeline catches drift in the columns of the data, you can use the `--drift` parameter to introduce a small amount of drift:

```shell
uv run pipelines/traffic.py run --drift True
```

This parameter will force one of the columns to be slightly different than the original dataset. This should be enough for the Monitoring pipeline to flag the column as having drifted from the reference data:

```python
if self.drift:
    rng = np.random.default_rng()
    self.data["body_mass_g"] += rng.uniform(
        1,
        3 * self.data["body_mass_g"].std(),
        size=len(self.data),
    )
```