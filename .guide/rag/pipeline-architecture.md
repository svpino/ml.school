# Pipeline Architecture

The [`Indexing`](src/pipelines/indexing.py) pipeline implements a straightforward linear flow with six steps. Each step builds upon the previous one to create a complete vector index:

```
start → prepare_documents → setup_embedding_model → create_vector_index → similarity_search → end
```

Let's examine each step in detail.

## Loading Documentation Files

The first step loads all documentation files from the specified directory. The pipeline processes both Markdown (`.md`) and Python (`.py`) files, as these typically contain the most valuable knowledge for RAG applications.

```python
@step
def start(self):
    """Load documentation from local directory."""
    directory = Path(self.location)

    if not directory.exists():
        msg = f"Directory not found: {directory}"
        raise FileNotFoundError(msg)
```

The pipeline recursively searches through the documentation directory and extracts metadata for each file:

- **File path**: Relative path for consistent identification
- **Content**: Full text content of the file
- **Section**: Top-level directory for categorization
- **Type**: File format (markdown or python)

This metadata enables sophisticated filtering during retrieval, allowing the RAG system to search within specific sections or file types when needed.

## Preparing Documents for Indexing

The second step transforms the loaded files into LangChain `Document` objects—the standard format for document processing in the LangChain ecosystem.

```python
@step
def prepare_documents(self):
    """Prepare the documents that we'll add to the vector store."""
    from langchain_core.documents import Document

    self.documents = [
        Document(
            page_content=d.content,
            metadata={"file": d.file, "section": d.section, "type": d.type},
        )
        for d in self.data.itertuples(index=False)
    ]
```

The pipeline also generates unique identifiers for each document using SHA-256 hashing of the file path. This ensures consistent document IDs across different pipeline runs, which is essential for updating existing indexes without duplicating content.

## Setting Up the Embedding Model

The third step initializes the embedding model that will convert text into vector representations. The pipeline uses a [custom embedding model](src/common/embeddings.py) built on top of LiteLLM, which provides a unified interface for various embedding providers.

```python
@step
def setup_embedding_model(self):
    """Initialize the embedding model we'll use to generate embeddings."""
    self.custom_embedding_model = CustomEmbeddingModel(self.embedding_model)

    self.embedding_dimensions = len(
        self.custom_embedding_model.embed_query("dimensions")
    )
```

The [`CustomEmbeddingModel`](src/common/embeddings.py) class implements the LangChain `Embeddings` interface and supports any embedding model available through LiteLLM. This flexibility allows you to experiment with different embedding models without changing the pipeline code.

The pipeline automatically detects the dimensionality of the chosen embedding model by generating a test embedding. This is crucial for configuring the vector index correctly, as different models produce embeddings of different sizes.

## Creating the Vector Index

The fourth step builds the actual vector index using Facebook AI Similarity Search (FAISS). FAISS provides efficient similarity search and clustering of dense vectors, making it ideal for RAG applications.

```python
@step
def create_vector_index(self):
    """Create the vector store and index the list of documents."""
    import faiss
    from langchain_community.vectorstores import FAISS

    self.vector_store = FAISS(
        embedding_function=self.custom_embedding_model,
        index=faiss.IndexFlatL2(self.embedding_dimensions),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    self.vector_store.add_documents(documents=self.documents, ids=self.ids)
```

The pipeline uses `IndexFlatL2`, which performs exact L2 (Euclidean) distance searches. While this approach is computationally expensive for large document collections, it provides the highest accuracy for similarity searches. For production systems with millions of documents, you might consider approximate algorithms like IVF or HNSW for better performance.

## Validating the Index

The fifth step performs a sample similarity search to validate that the index works correctly. This serves as both a quality check and a demonstration of the index's capabilities.

```python
@step
def similarity_search(self):
    """Perform a similarity search to ensure everything works."""
    query = "branches"

    results = self.vector_store.similarity_search(
        query,
        k=2,
        filter={"type": "markdown"},
    )
```

The validation step searches for documents related to "branches" and filters results to include only Markdown files. This demonstrates two key RAG capabilities:

- **Semantic search**: Finding documents based on meaning rather than exact keyword matches
- **Metadata filtering**: Restricting searches to specific document types or sections

## Persisting the Index

The final step saves the vector index to disk for use by RAG applications. The pipeline organizes indexes by embedding model name, allowing you to maintain multiple indexes with different embedding models simultaneously.

```python
@step
def end(self):
    """Save the vector store to a local directory."""
    self.vector_store.save_local(f"data/index/{self.embedding_model}")
```

This organization is valuable for experimentation and A/B testing different embedding models in your RAG system.