# Assignments

Complete the following assignments to reinforce the concepts we covered in this session. You don't have to work on every assignment. Pick the ones that you think will help you the most.

1. The [`backend`](src/inference/backend.py) implementation in the codebase supports storing the input data and predictions in an SQLite database. Create a new backend that supports storing the data in a file using the [JSON Lines](https://jsonlines.org/) text format. Each entry in the file should correspond to an input request and a prediction.

1. Create a new [`backend`](src/inference/backend.py) implementation that stores predictions and inputs in a CSV file format. The backend should include timestamp information for each new request and prediction.

1. Add logging capabilities to the [Inference pipeline](src/inference/model.py) to record prediction latency and any errors that occur.

1. Implement caching functionality in the [Inference pipeline](src/inference/model.py) to store recent predictions in memory using a hash of the input data. If the same input is received again, return the cached prediction instead of recomputing it.

1. Create a confidence threshold mechanism where the [Inference pipeline](src/inference/model.py) only returns predictions when the model's confidence is above a specified threshold. For low-confidence predictions, return an "uncertain" response with the confidence score.

1. Use a Large Language Model to suggest what additional unit tests would be necessary to increase the test coverage of the inference pipeline.

1. Modify the [Inference pipeline](src/inference/model.py) to add input validation that checks for required fields, validates data types, and ensures numerical values are within reasonable ranges (e.g., the body mass of the penguin must be between 1000 grams and 10000 grams). The pipeline should return descriptive error messages for invalid inputs.

1. If the [Inference pipeline](src/inference/model.py) receives a batch of samples and one contains an error, the pipeline doesn't process any of the samples in the batch. Modify the code so the pipeline processes all the samples except the one with the error.

1. Create a custom preprocessing step in the [Inference pipeline](src/inference/model.py) that can handle different input formats (JSON, CSV, XML) and automatically detect the format. The pipeline should convert all formats to the standard pandas DataFrame format before processing.

1. Implement model versioning support in the [Inference pipeline](src/inference/model.py) that allows the pipeline to load and serve different versions of the model based on request headers or configuration.