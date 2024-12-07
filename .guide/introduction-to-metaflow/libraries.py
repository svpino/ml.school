from metaflow import FlowSpec, conda, conda_base, step


@conda_base(python="3.12.8", packages={"pandas": "2.2.3"})
class Libraries(FlowSpec):
    """A flow that showcases how to manage libraries."""

    @conda(packages={"matplotlib": "3.9.3"})
    @step
    def start(self):
        """Import and print the version of Matplotlib."""
        import matplotlib as mpl

        print("Matplotlib version:", mpl.__version__)
        self.next(self.end)

    @step
    def end(self):
        """Import global and local libraries again."""
        # Pandas is available here because we installed it using the @conda_base
        # decorator.
        import pandas as pd

        print("Pandas version:", pd.__version__)

        # Matplotlib is not available here because we only made it available in
        # the `start` step.
        try:
            import matplotlib as mpl

            print("Matplotlib version:", mpl.__version__)
        except ImportError:
            print("Matplotlib not installed")


if __name__ == "__main__":
    Libraries()
