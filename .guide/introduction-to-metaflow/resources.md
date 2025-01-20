# Compute Resources

Metaflow can handle tasks requiring any [computational power](https://docs.metaflow.org/scaling/remote-tasks/requesting-resources). Metaflow allocates around 1 CPU core and 4GB of RAM to each flow step by default. If a step requires more resources, such as additional CPU cores, memory, disk space, or GPUs, you can specify these needs using the `@resources` decorator. 

The `@resources` decorator has no effect when running a flow locally, but when running a flow in a remote computing environment, it will instruct Metaflow to provision instances with sufficient resources to execute the step.

In a production environment where all runs occur in the cloud, the `@resources` decorator will ensure that resource needs are respected, enabling your flows to scale efficiently for larger datasets or more complex computations. 

In the example code, the `matrix` step generates a matrix that requires 8,192 MB of memory. To ensure this step runs in a production environment, we can use the `@resources` decorator to request the appropriate amount of RAM.

Run the following command in the terminal to execute the flow:

```shell
uv run -- python .guide/introduction-to-metaflow/resources.py \
    --environment conda run
```