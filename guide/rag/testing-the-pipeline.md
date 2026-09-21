# Testing the Pipeline

You can run the tests associated with the indexing functionality by executing the following command:

```shell
uv run pytest -k test_indexing
```

The tests verify that the pipeline correctly loads documents, generates embeddings, creates the vector index, and performs similarity searches as expected.