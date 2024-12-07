# Parameterizing Flows

You can [parameterize](https://docs.metaflow.org/api/flowspec#parameters) a flow by assigning a `Parameter` object to a class variable. Metaflow will make parameters available at every step in the flow as instance variables.

In the example code, you have two parameters `one` and `two`. The first parameter is an integer, and the second is a string. Metaflow determines the type of a parameter based on its default value. You can also specify the type of a parameter explicitly by passing the `type` argument to the `Parameter` constructor:

```python
three = Parameter("three", help="Third parameter", type=float)
```

You can set a parameter's value on the command line when running the flow by passing each parameter as keyword arguments to the `run` command:

```shell
uv run -- python .guide/introduction-to-metaflow/parameters.py run \
    --one 10 --two Twenty
```

Metaflow will use the default values specified in the flow definition if the parameter values are not provided from the command line. The following command will run the flow with the default values:

```shell
uv run -- python .guide/introduction-to-metaflow/parameters.py run
```

If a parameter is required and it's not provided from the command line, Metaflow will raise an error.