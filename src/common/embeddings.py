import litellm
from langchain_core.embeddings import Embeddings


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
