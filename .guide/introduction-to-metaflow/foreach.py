from metaflow import FlowSpec, step


class Foreach(FlowSpec):
    """A flow that showcases how the foreach works."""

    @step
    def start(self):
        """Initialize the start value artifact."""
        self.people = ["alice", "bob", "charlie"]
        self.next(self.capitalize, foreach="people")

    @step
    def capitalize(self):
        """Capitalize the input name."""
        name = self.input or ""
        self.person = name.capitalize()
        print(f'Turned "{name}" into "{self.person}"')

        self.next(self.join)

    @step
    def join(self, inputs):
        """Join the results of the foreach."""
        self.people = [i.person for i in inputs]
        self.next(self.end)

    @step
    def end(self):
        """Print the final list of people."""
        print("People:", self.people)


if __name__ == "__main__":
    Foreach()
