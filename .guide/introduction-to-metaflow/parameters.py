from metaflow import FlowSpec, IncludeFile, Parameter, step


class Parameters(FlowSpec):
    """A flow that showcases how to use parameters."""

    one = Parameter("one", help="First parameter", default=1)
    two = Parameter("two", help="Second parameter", default="two")

    file = IncludeFile(
        "file",
        is_text=True,
        help="Sample comma-separated file",
    )

    @step
    def start(self):
        """Print the initial value of the parameters."""
        print("Parameter one:", self.one)
        print("Parameter two:", self.two)
        self.next(self.end)

    @step
    def end(self):
        """Print the contents of the included file."""
        print("Included file:")
        print(self.file)


if __name__ == "__main__":
    Parameters()
