import hashlib
from pathlib import Path

import litellm
import pandas as pd
from common import Pipeline
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from metaflow import Parameter, step

# TODO:
# 1. parametrizar el location del vector store
# 2. hacer que los env vars estén disponibles a traves del @environment
# 3. Mirar aqui como se hace los @environment y hacerlo https://github.com/outerbounds/rag-demo/blob/main/flows/pinecone_index.py
# 4. Probar un embedding model diferente
# 5. Necesito un nuevo script para chatear usando el vector store. Maybe un MCP?


class CustomEmbeddingModel(Embeddings):
    """Custom text embedding implementation model.

    This is the implementation of the `Embeddings` interface to map text to vectors.
    This implementation uses LiteLLM to allow flexible model selection to generate
    embeddings.
    """

    def __init__(self, model: str) -> None:
        """Initialize the embedding model."""
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed the supplied list of documents."""
        response = litellm.embedding(model=self.model, input=texts)
        return [d["embedding"] for d in response["data"]]

    def embed_query(self, text: str) -> list[float]:
        """Embed the supplied query text."""
        return self.embed_documents([text])[0]


class Indexing(Pipeline):
    """A Metaflow pipeline used for indexing the documentation of the project.

    This pipeline implements the necessary steps to load, process, and index
    the documentation of the project for efficient retrieval.
    """

    location = Parameter(
        "location",
        help="The location of the documentation files.",
        default=".guide/",
    )

    embedding_model = Parameter(
        "embedding-model",
        help="The model to use for generating embeddings.",
        default="gemini/text-embedding-004",
    )

    @step
    def start(self):
        """Load documentation from local directory."""
        directory = Path(self.location)

        if not directory.exists():
            msg = f"Directory not found: {directory}"
            raise FileNotFoundError(msg)

        files: list[dict[str, str]] = []

        # Let's list every file in the documentation directory and filter
        # by file type before processing them.
        for f in directory.rglob("*"):
            # We only want to process Markdown and Python files.
            if f.is_file() and f.suffix in (".md", ".py"):
                text = f.read_text(encoding="utf-8")

                relative_path = f.relative_to(directory)
                parts = relative_path.parts
                section = parts[0] if len(parts) > 1 else ""

                files.append(
                    {
                        "file": str(relative_path),
                        "content": text,
                        "section": section,
                        "type": "markdown" if f.suffix == ".md" else "python",
                    }
                )

        files.sort(key=lambda r: r["file"])
        self.data = pd.DataFrame(files, columns=list(files[0].keys()))

        self.logger.info("Number of files: %d", len(self.data))

        self.next(self.prepare_documents)

    @step
    def prepare_documents(self):
        """Prepare the documents that we'll add to the vector store."""
        from langchain_core.documents import Document

        # Let's go through every entry in the DataFrame and create a Document object
        # with the content of the file and the corresponding metadata.
        self.documents = [
            Document(
                page_content=d.content,
                metadata={"file": d.file, "section": d.section, "type": d.type},
            )
            for d in self.data.itertuples(index=False)
        ]

        # To index the documents in the vector store, we need to generate unique
        # identifiers for each document. We can use the file path for this purpose
        # to ensure these identifiers are consistent across different runs.
        self.ids = [
            hashlib.sha256(f.encode("utf-8")).hexdigest()
            for f in self.data["file"].tolist()
        ]

        self.logger.info("Documents prepared: %d", len(self.documents))

        self.next(self.setup_embedding_model)

    @step
    def setup_embedding_model(self):
        """Initialize the embedding model we'll use to generate embeddings."""
        self.logger.info("Embedding model: %s", self.embedding_model)

        # We'll use a custom embedding model to generate embeddings
        # using LiteLLM.
        self.custom_embedding_model = CustomEmbeddingModel(self.embedding_model)

        # Since we don't know beforehand which embedding model we'll be using,
        # let's infer the dimensions by generating an embedding and checking
        # its length.
        self.embedding_dimensions = len(
            self.custom_embedding_model.embed_query("dimensions")
        )

        self.logger.info("Embedding dimensions: %d", self.embedding_dimensions)

        self.next(self.create_vector_index)

    @step
    def create_vector_index(self):
        """Create the vector store and index the list of documents."""
        import faiss
        from langchain_community.docstore.in_memory import InMemoryDocstore
        from langchain_community.vectorstores import FAISS

        self.logger.info("Creating FAISS vector store...")

        # Let's create a FAISS vector store using the custom embedding model
        # and the dimension of the index.
        self.vector_store = FAISS(
            embedding_function=self.custom_embedding_model,
            index=faiss.IndexFlatL2(self.embedding_dimensions),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        # Now, we can add the list of documents we prepared before.
        self.vector_store.add_documents(documents=self.documents, ids=self.ids)

        self.next(self.similarity_search)

    @step
    def similarity_search(self):
        """Perform a similarity search to ensure everything works."""
        query = "branches"
        self.logger.info('Similarity search: "%s"', query)

        # Let's perform the similarity search and return the top 2 Markdown documents.
        results = self.vector_store.similarity_search(
            query,
            k=2,
            filter={"type": "markdown"},
        )

        for result in results:
            self.logger.info(
                "• File: %s. Section: %s. Type: %s.",
                result.metadata["file"],
                result.metadata["section"],
                result.metadata["type"],
            )

        self.next(self.end)

    @step
    def end(self):
        """Save the vector store to a local directory."""
        self.vector_store.save_local("data/index")

        self.logger.info("Indexing complete.")


if __name__ == "__main__":
    load_dotenv(override=True)
    Indexing()
