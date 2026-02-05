# Retrieval-Augmented Generation

Retrieval-Augmented Generation combines the power of large language models with external knowledge retrieval. The indexing process is the foundation of any RAG system. It transforms your documents into a searchable format that allows the system to find relevant information to augment the model's responses.

Traditional language models are limited by their training data cutoff and cannot access real-time information or domain-specific knowledge that wasn't included during training. RAG solves this limitation by retrieving relevant context from external knowledge bases and injecting it into the model's prompt. This approach enables the model to provide accurate, up-to-date, and contextually relevant responses.

The Indexing pipeline processes documentation files and creates a searchable vector index for Retrieval-Augmented Generation (RAG) applications. Every time it runs, the pipeline loads documentation from a specified directory, generates embeddings, and builds a FAISS vector store that enables efficient semantic search.

![Indexing pipeline](.guide/rag/images/indexing.png)

To run the [Indexing pipeline](src/pipelines/indexing.py) locally, execute the command below:

```shell
uv run src/pipelines/indexing.py run
```

The Indexing pipeline loads documentation files from the `.guide/` directory by default, processes both Markdown and Python files, generates embeddings using a configurable model, and creates a FAISS vector index. After running the pipeline, you should see a new vector index saved in the `data/index/` directory.

The pipeline supports configurable parameters for both the documentation location and the embedding model. You can specify different values using the following command:

```shell
uv run src/pipelines/indexing.py run \
    --location docs/ \
    --embedding-model gemini/text-embedding-004
```

To display the supported parameters of the Indexing pipeline, run the following command:

```shell
uv run src/pipelines/indexing.py run --help
```

You can observe the execution of the pipeline and visualize its results by running a Metaflow card server using the following command:

```shell
uv run src/pipelines/indexing.py card server
```

After the card server is running, open your browser and navigate to [localhost:8324](http://localhost:8324/). Every time you run the Indexing pipeline, the viewer will automatically update to show the cards related to the latest pipeline execution.

# Configuration Options

The Indexing pipeline supports two main configuration parameters:

- **`location`**: The directory containing documentation files (default: `.guide/`)
- **`embedding-model`**: The model used for generating embeddings (default: `gemini/text-embedding-004`)

You can experiment with different embedding models by specifying alternative models supported by LiteLLM:

```shell
# Using OpenAI's embedding model
uv run src/pipelines/indexing.py run \
    --embedding-model text-embedding-3-small

# Using a different Google model
uv run src/pipelines/indexing.py run \
    --embedding-model gemini/text-embedding-preview-0409
```