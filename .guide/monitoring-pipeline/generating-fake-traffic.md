# Generating Fake Traffic

Before running the Monitoring pipeline, we'll generate some traffic for the hosted model. We wouldn't need this for a production model that receives live traffic from users, but the fake traffic works for testing purposes.

To simplify the process, you can send multiple requests to your hosted model using the Traffic pipeline:

```shell
uv run -- python pipelines/traffic.py \
    --environment=conda run
```

You can also run the pipeline using `just` along with the `traffic` recipe:

```shell
just traffic
```

This pipeline loads the original dataset, randomly selects a number of samples, and sends them to the hosted model in batches. 

Using the `--endpoint` parameter, you can specify how to communicate with the hosted model. This parameter expects the name of a class implementing the [`endpoint.Endpoint`](pipelines/inference/endpoint.py) abstract class. By default, this parameter will use the [`endpoint.Server`](pipelines/inference/endpoint.py) implementation, which knows how to submit requests to an inference server created using the `mlflow models serve` command.

To specify the location of the hosted model, you can use the `--target` parameter. By default, the pipeline assumes you are running the model locally, on port `8080`, on the same computer from where you are running the Traffic pipeline:

```python
target = Parameter(
    "target",
    default="http://127.0.0.1:8080/invocations",
)
```

By default, the Traffic pipeline will send 200 samples to the hosted model. If you want to send a different number, use the `--samples` parameter:

```shell
uv run -- python pipelines/traffic.py \
    --environment=conda run \
    --samples 500
```

Finally, to evaluate whether the Monitoring pipeline catches drift in the columns of the data, you can use the `--drift` parameter to introduce a small amount of drift in of the columns:

```shell
uv run -- python pipelines/traffic.py \
    --environment=conda run \
    --drift True
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