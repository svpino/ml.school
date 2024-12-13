# Parallel Branches

In Metaflow, you can execute steps in parallel using [branches](https://docs.metaflow.org/metaflow/basics#branch). Branches can improve the performance of a flow by utilizing multiple CPU cores or cloud instances.

Every branch must eventually converge at a `join` step, where Metaflow will combine the outputs from the parallel steps. To understand how the data propagates through a flow and parallel branches, check "[Data flow through the graph](https://docs.metaflow.org/metaflow/basics#data-flow-through-the-graph)."

In the example code, the flow begins with the `start` step, which branches into two parallel steps: `step1` and `step2`. Both branches then converge at the `join` step.

![Parallel branches](.guide/introduction-to-metaflow/images/branches.png)

Notice how the code sets the `common` artifact in both branches to different values, making it ambiguous when we reach the `join` step. Metaflow provides the [`merge_artifacts()`](https://docs.metaflow.org/api/flowspec#FlowSpec.merge_artifacts) function to help propagate values in a `join` step. In this case, the function propagates the `start_value` artifact and excludes the ambiguous `common` artifact.

When you have ambiguous artifacts, you can reference them using the name of the step. For example, the `join` step accesses `inputs.step1.common` and `inputs.step2.common` to print their values.

Finally, notice you can also iterate over all steps in the branch using the `inputs` parameter to gain access to every artifact.

Run the following command in the terminal to execute the flow:

```shell
uv run -- python .guide/introduction-to-metaflow/branches.py run
```
