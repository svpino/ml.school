from metaflow import FlowSpec, conda_base, resources, step


@conda_base(python="3.12.8", packages={"numpy": "2.2.2"})
class Resources(FlowSpec):
    """A flow that showcases how to use request computing resources."""

    @step
    def start(self):
        """Compute the dimensions of the matrix."""
        import numpy as np

        # We want to create a matrix that requires 1024 MB of memory, so let's
        # calculate the number of elements that will fit in that memory.
        memory = 1024 * 1024 * 1024
        dtype = np.float64
        elements = memory // np.dtype(dtype).itemsize

        # We can now determine how many rows and columns will fit in that memory.
        self.rows = self.columns = int(elements**0.5)

        print(f"Matrix dimensions: {self.rows} x {self.columns}")

        self.next(self.matrix)

    @resources(cpu=1, memory=1024)
    @step
    def matrix(self):
        """Generate a random matrix and sum its values."""
        import numpy as np

        rng = np.random.default_rng()
        matrix = rng.random((self.rows, self.columns))
        print(f"Memory used (MB): {matrix.nbytes / (1024 ** 2):.2f}")

        self.value = np.sum(matrix).item()

        self.next(self.end)

    @step
    def end(self):
        """Print the final value."""
        print(f"Final result: {self.value:.2f}")


if __name__ == "__main__":
    Resources()
