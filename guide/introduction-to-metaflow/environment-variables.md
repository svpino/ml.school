# Environment Variables

To access [environment variables](https://docs.metaflow.org/api/step-decorators/environment) from a flow, you can use the `@environment` Metaflow decorator. This decorator works when Metaflow runs locally and on a remote computing environment.

The example code defines an environment variable `VARIABLE` and accesses it in the `start` step. Notice how the value of this variable comes from the environment variable `METAFLOW_VARIABLE`.

The variable `METAFLOW_VARIABLE` would be available directly when running the flow locally, but it wouldn't be available when running the flow on a remote compute instance. The `@environment` decorator allows us to access the value of `METAFLOW_VARIABLE` in both cases.

Before running the flow, you must set the environment variable `METAFLOW_VARIABLE`. You can do that in a single command:

```shell
METAFLOW_VARIABLE=123 \
    uv run .guide/introduction-to-metaflow/src/environment.py run
```

If you run the flow without setting `METAFLOW_VARIABLE`, the value of `VARIABLE` will be `None` because it won't be able to access the environment variable. Run the following command and check the output:

```shell
uv run .guide/introduction-to-metaflow/src/environment.py run
```
