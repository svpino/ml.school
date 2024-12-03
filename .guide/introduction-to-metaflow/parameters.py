from metaflow import FlowSpec, Parameter, step


class Parameters(FlowSpec):
    """A flow that showcases how to use parameters."""

    one = Parameter("one", help="First parameter", default=1)
    two = Parameter("two", help="Second parameter", default="two")

    @step
    def start(self):
        """Print the initial value of the parameters."""
        print("Parameter one:", self.one)
        print("Parameter two:", self.two)
        self.next(self.end)

    @step
    def end(self):
        """End of the flow."""


if __name__ == "__main__":
    Parameters()
