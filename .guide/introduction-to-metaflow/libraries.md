# Managing Libraries

Metaflow makes it easy to [manage external dependencies](https://docs.metaflow.org/scaling/dependencies/libraries) using the `@conda` and `@pypi` decorators. These decorators allow us to specify the packages required to run each step, ensuring isolated and reproducible environments when running a flow locally or in a remote computing environment.

The `@conda` and `@pypi` decorators support any libraries as if you were installing them manually with `conda` or `pip`. You must specify the version of each library you want to import, ensuring the code executes correctly even when the packages change over time.

When running a flow that uses `@conda` or `@pypi`, the flow won't have access to any packages installed at the system level. Metaflow ensures that the environment is fully self-contained and the decorators manage every dependency explicitly.

To apply the same environment across all steps in a flow, you can use the `@conda_base` or `@pypi_base` decorators at the class level, eliminating redundancy.

In the example code, we make the `pandas` library available for the entire flow and the `matplotlib` library available only for the `start` step.

When running a flow for the first time, Metaflow resolves and creates the environment, which may take a few minutes. Metaflow uses the cached environment for subsequent runs, making the startup time significantly faster.

The main advantage of managing libraries using these decorators is the stability in production environments. Once defined, these environments remain consistent regardless of changes in package versions or availability, ensuring predictable and reliable execution for production runs.

Whenever you use the `@conda` or `@pypi` decorators, you must specify the `--environment` option when running the flow:

```shell
uv run -- python .guide/introduction-to-metaflow/libraries.py \
    --environment conda run
```
