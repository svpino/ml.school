from metaflow import step

from pipelines.common import Pipeline, backend, dataset


class MockPipeline(Pipeline):
    """A mock pipeline for testing purposes."""

    @dataset
    @backend
    @step
    def start(self):  # noqa: D102
        self.next(self.end)

    @step
    def end(self):  # noqa: D102
        pass


if __name__ == "__main__":
    MockPipeline()
