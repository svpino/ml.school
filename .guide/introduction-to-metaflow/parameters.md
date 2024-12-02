# Parameterizing Flows

You can [parameterize](https://docs.metaflow.org/api/flowspec#parameters) a flow by assigning a `Parameter` object to a class variable. Metaflow will make paramaters available to every step in the flow as attributes of the flow object.

In the example code, you have two parameters `one` and `two`. The first parameter is an integer and the second parameter is a string. Metaflow determines the type of a parameter based on their default value. You can also specify the type of a parameter explicitly by passing the `type` argument to the `Parameter` constructor:

```python
three = Parameter("three", help="Third parameter", type=float)
```

You can set the value of a parameter on the command line when running the flow. The parameter values are passed to the flow as keyword arguments to the `run` method.

```bash
python parameters.py run --one 10 --two Twenty
```

If the parameter values are not provided on the command line, Metaflow will use the default values specified in the flow definition. If there are no default values, Metaflow will raise an error when running the flow.

You can also [include files](https://docs.metaflow.org/scaling/data#data-in-local-files) as part of your flow using the same mechanism. Metaflow will version these files and make them accessible to all the steps in the flow.

In the example code, we include a text file named `file` as part of the flow. If specified from the command line, Metaflow will make the contents of the file available to every step of the flow:

```bash
python parameters.py run --file path/to/sample.csv
```
