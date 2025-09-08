# Assignments

Complete the following assignments to reinforce the concepts we covered in this session. You don't have to work on every assignment. Pick the ones that you think will help you the most.

1. The [Metaflow documentation](https://docs.metaflow.org/) is an excellent resource for understanding the library. Spend some time browsing the documentation. It has excellent examples, and you'll get a better appreciation for everything you can do with the library.

1. Create a simple flow that tracks a sequence of numerical operations. In the first step, initialize an artifact with a number. In each subsequent step, update the artifact by applying a different arithmetic operation (e.g., addition, subtraction, multiplication) and append each new value to a list. In the final step, print the entire history of values and calculate both the sum and average.

1. Create a flow that starts by initializing an artifact with a numerical value. Then split into two predetermined parallel branches, where the first branch adds a constant to the artifact and the second branch multiplies the artifact by a constant. In a subsequent join step, merge the results by printing both branch outcomes and computing the sum of the two outcomes.

1. Create a flow that takes a list of numbers as a parameter. Use a foreach loop to square each number in a separate step. In the join step, collect the squared results and print both the full list and the total sum.

1. Add a `@retry` decorator to a step in your flow that simulates a flaky external service (e.g., randomly fails 50% of the time). Demonstrate that the step will automatically retry a few times before failing.

1. Design a custom card that goes beyond static HTML and integrates some data visualization. Implement a flow that generates a random dataset and use the `@card` decorator to show charts or tables related to the data.

1. Create a flow that compares the use of the `@environment` decorator with accessing environment variables using the `python-dotenv` library to load environment variables from a `.env` file.

1. Create a flow that includes an external CSV file using the `IncludeFile` function. In a processing step, parse the file and print the number of rows and columns. Add error handling to check for issues such as an empty file or malformed content, and print an appropriate error message if the file cannot be processed.

1. Write a flow that takes a text prompt and uses an LLM to generate a short response. Store the response as an artifact and create a custom card that visualizes the prompt/response pair nicely in a styled format.

1. Create a flow that loads a JSON configuration file using the `Config` object. Use `Parameter` objects as part of the flow to override some of the configuration values at runtime. Demonstrate how the flow behaves differently based on the configuration and parameter inputs.

1. Design a flow with parallel branches that have conflicting artifact names. Use the `merge_artifacts()` function.

1. Create a flow with three parallel branches to process different aspects of the same dataset. The first branch will calculate the mean of a column, the second branch will calculate the median, and the third branch will calculate the standard deviation. Each branch will use overlapping artifact names (e.g., `result`). In the join step, use `merge_artifacts()` to handle conflicts appropriately and access branch-specific results using the `inputs` parameter. Show how to aggregate results from all branches while preserving branch-specific computations.

1. Create a flow that starts by generating a list of dictionaries using an LLM. Each item of the list will represent a student, and each student will have a name and score. The flow should use a foreach loop to process each student on a separate branch. Each branch will transform the student's name to uppercase and increase the score by a fixed amount (e.g., add 10). In the join step, the flow will aggregate all of the scores and print both the updated dictionaries and the aggregate result. 

