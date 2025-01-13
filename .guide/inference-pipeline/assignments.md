# Assignments

Complete the following assignments to reinforce the concepts we covered in this session. You don't have to work on every assignment. Pick the ones that you think will help you the most.

1. The backend implementation in the codebase supports storing the input data and predictions in an SQLite database. Create a new backend that supports storing the data in a file using the JSON Lines format. Each entry in the file should correspond to an input request and a prediction.

1. If the inference pipeline receives a batch of samples and one contains an error, the pipeline doesn't process any of the samples in the batch. Modify the code so the pipeline processes all the samples except the one with the error.

1. Use a Large Language Model to suggest what additional unit tests would be necessary to increase the test coverage of the inference pipeline.
 