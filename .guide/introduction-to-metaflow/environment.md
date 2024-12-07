# Environment Variables

To access [environment](https://docs.metaflow.org/api/step-decorators/environment) variables from a flow, you can use the `@environment` Metaflow decorator. This decorator works when Metaflow runs locally and on a remote computing environment.

The example code defines an environment variable `VARIABLE` and accesses it in the `start` step. Notice how the value of this variable comes from the environment variable `METAFLOW_VARIABLE`.

The variable `METAFLOW_VARIABLE` would be available directly when running the flow locally, but it wouldn't be available when running the flow on a remote compute instance. The `@environment` decorator allows us to access the value of `METAFLOW_VARIABLE` in both cases.

Before running the flow, you must set the environment variable `METAFLOW_VARIABLE`. You can do that in a single command:

```shell
METAFLOW_VARIABLE=123 \
    uv run -- python .guide/introduction-to-metaflow/environment.py run
```