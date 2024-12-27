from metaflow import FlowSpec, project, step

from pipelines.common import DatasetMixin


@project(name="penguins")
class TestFlowMixinFlow(FlowSpec, DatasetMixin):
    """Pipeline used to test the FlowMixin class."""

    @step
    def start(self):  # noqa: D102
        self.data = self.load_dataset()
        self.next(self.end)

    @step
    def end(self):  # noqa: D102
        pass


if __name__ == "__main__":
    TestFlowMixinFlow()
