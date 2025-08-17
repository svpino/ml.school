from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from metaflow import FlowSpec, project, step


def load_guide_markdown(base_dir: str | Path = ".guide") -> pd.DataFrame:
    """Return a DataFrame with all markdown and Python files under `base_dir`.

    Columns
    -------
    File path: path of the file relative to the guide root.
    Content: full text of the file.
    section: top-level folder under the guide root where the file is located
        (empty string when the file is at the root of `base_dir`).
    """
    base = Path(base_dir)
    columns = ["file", "content", "section"]
    if not base.exists():
        return pd.DataFrame(columns=columns)

    exts = {".md", ".py"}
    rows: list[dict[str, str]] = []
    for p in base.rglob("*"):
        if p.is_file() and p.suffix in exts:
            try:
                text = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = p.read_text(encoding="utf-8", errors="replace")
            rel = p.relative_to(base)
            parts = rel.parts
            section = parts[0] if len(parts) > 1 else ""
            rows.append({
                "file": str(rel),
                "content": text,
                "section": section,
            })
    rows.sort(key=lambda r: r["file"])  # deterministic ordering
    return pd.DataFrame(rows, columns=columns)


@project(name="mlschool")
class Indexing(FlowSpec):
    """A simple Metaflow pipeline used for indexing guide markdown files."""

    @step
    def start(self):
        """Start the flow."""
        print("Start")
        df = load_guide_markdown(".guide/")
        print(df)
        self.next(self.vs)

    @step
    def vs(self):
        import faiss
        from langchain_community.docstore.in_memory import InMemoryDocstore
        from langchain_community.vectorstores import FAISS
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        # https://ai.google.dev/gemini-api/docs/embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001")

        index = faiss.IndexFlatL2(3072)

        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        # need to generate ID of document based on file path

        # vector_store.save_local("index")

        self.next(self.end)

    @step
    def end(self):
        """End the flow."""
        print("End")


if __name__ == "__main__":
    load_dotenv(override=True)
    Indexing()
