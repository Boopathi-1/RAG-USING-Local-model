from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer


class LocalEmbedding:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()


def get_vector_store(documents=None, persist_dir="data/chroma_db"):
    embedding = LocalEmbedding()

    if documents:
        db = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=persist_dir
        )
        

    else:
        db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding
        )

    return db

