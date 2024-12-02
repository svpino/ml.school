from metaflow import FlowSpec, card, step


class Cards(FlowSpec):
    """A flow that showcases how to use cards to visualize results."""

    @card
    @step
    def start(self):
        """Initialize a few artifacts."""
        self.orange = 1
        self.apple = 2

        self.next(self.report)

    @card(type="html")
    @step
    def report(self):
        """Generate a custom card."""
        self.html = f"""
        <h1>Custom Metaflow Card</h1>
        <p style="background: orange; padding: 8px; width: 300px;">
            Orange: {self.orange}
        </p>
        <p style="background: red; padding: 8px; width: 300px;">
            Apple: {self.apple}
        </p>
        """
        self.next(self.end)

    @card
    @step
    def end(self):
        """Compute new artifact."""
        self.juice = self.orange + self.apple


if __name__ == "__main__":
    Cards()
