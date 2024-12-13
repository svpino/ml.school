# Metaflow Artifacts

Metaflow uses [artifacts](https://docs.metaflow.org/metaflow/basics#artifacts) to track the values assigned to instance variables within a flow.

Artifacts simplify managing the flow of data through a workflow. Metaflow automatically persists artifacts and carries their value from one step to another. You can access the value of these artifacts at any point for debugging and analysis purposes.

Artifacts behave consistently across different environments, whether you run steps locally or in a remote computing environment.

In the example code, `self.variable` is an artifact. The flow initializes it in the `start` step, increments it in the `increment` step, and finally prints it in the `end` step.

![Metaflow artifacts](.guide/introduction-to-metaflow/images/artifacts.png)

If you run the `increment` step remotely and the `end` step locally, Metaflow will ensure that the artifact's updated value stays consistent and moves through the steps correctly.

Run the following command in the terminal to execute the flow:

```shell
uv run -- python .guide/introduction-to-metaflow/artifacts.py run
```
