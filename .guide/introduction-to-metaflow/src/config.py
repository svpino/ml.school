import yaml
from metaflow import Config, FlowSpec, step


class Configuration(FlowSpec):
    """A flow that showcases how to use config files."""

    config = Config(
        "config",
        default=".guide/introduction-to-metaflow/data/config1.yml",
        parser=yaml.full_load,
    )

    @step
    def start(self):
        """Print configuration file."""
        print("Configuration:", self.config)
        self.next(self.end)

    @step
    def end(self):
        """Print individual configuration values."""
        print("Value 1:", self.config.value1)
        print("Value 2:", self.config.value2)


if __name__ == "__main__":
    Configuration()
