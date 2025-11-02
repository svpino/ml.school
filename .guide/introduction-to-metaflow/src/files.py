from metaflow import FlowSpec, IncludeFile, step


class Files(FlowSpec):
    """A flow that showcases how to include a file in a flow."""

    file = IncludeFile(
        "file",
        is_text=True,
        help="Sample comma-separated file",
    )

    @step
    def start(self):
        """Print the contents of the included file."""
        print("Included file:")
        print(self.file)

        self.next(self.end)

    @step
    def end(self):
        """End the flow."""


if __name__ == "__main__":
    Files()
