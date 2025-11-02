# Configuring Flows

You can configure the behavior of a Metaflow flow using the [`Config`](https://docs.metaflow.org/metaflow/configuring-flows/introduction) object.

The `Config` object allows you to set configuration options that affect the entire flow. You can use these configuration settings to configure steps, flow decorators, and set the default values for your flow [Parameters](.guide/introduction-to-metaflow/parameterizing-flows.md).

The most basic use of the `Config` object is to read configuration settings from a JSON file. Imagine you have a file `config.json` with the following content:

```json
{
    "value1": 100,
    "value2": 200
}
```

You can read this configuration file from your flow by writing the following code:

```python
config = Config(
    "configuration",
    default="config.json"
)
```

You can now reference any of the configuration values in any step of the flow:

```python
@step
def start(self):
    print("Value1:", self.config.value1)
    self.next(self.end)
```

You can also use custom parsers to load and process configuration files. The example code corresponding to this section demonstrates how to implement a custom parser for a YAML configuration file:

```python
import yaml

...

config = Config(
    "config",
    default=".guide/introduction-to-metaflow/data/config1.yml",
    parser=yaml.full_load,
)
```

To execute the pipeline with the default config file, run the following command:

```shell
uv run .guide/introduction-to-metaflow/src/config.py run
```

You can also specify a different config file at runtime using the `--config` option, which takes the name of the `Config` property in the flow and the corresponding config file:

```shell
uv run .guide/introduction-to-metaflow/src/config.py \
    --config config .guide/introduction-to-metaflow/data/config2.yml run
```

Finally, instead of a file, you can pass the full config on the command line using the `--config-value` option:

```shell
uv run .guide/introduction-to-metaflow/src/config.py \
    --config-value config '{"value1": 1000, "value2": 2000}' run
```