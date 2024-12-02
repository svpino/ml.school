from metaflow import FlowSpec, card, step


class Cards(FlowSpec):
    """A flow that showcases how to use request computing resources."""

    @card
    @step
    def start(self):
        """Compute the dimensions of the matrix."""
        self.orange = 1
        self.apple = 2

        self.next(self.end)

    @card
    @step
    def end(self):
        """Print the final value."""
        self.juice = self.orange + self.apple


if __name__ == "__main__":
    Cards()
