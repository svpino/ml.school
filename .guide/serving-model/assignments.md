# Assignments

Complete the following assignments to reinforce the concepts we covered in this session. You don't have to work on every assignment. Pick the ones that you think will help you the most.

1. Implement a simple Python script that sends a batch of sample requests to a model running locally. The script should select a few random samples from the `data/penguins.csv` dataset, format them as JSON, and send them to the model server using an HTTP POST request.

1. Run the model server using MLServer[MLServer](https://mlserver.readthedocs.io/en/latest/) instead of the default Flask server.

1. Create a simple backend implementation that stores requests and predictions in [JSON Lines](http://jsonlines.org/) text format instead of SQLite. Run the model server using this new backend and test it by sending sample requests.

1. Create a script that starts two instances of the model server on different ports, each serving a different version of the model. Build a simple load balancer that routes requests between the two servers to perform A/B testing of model versions.

1. Create a load testing script that sends multiple concurrent requests to the local model server to evaluate its performance under load. Measure and report metrics like average response time, throughput, and error rates.