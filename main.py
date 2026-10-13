from ingestion.parser import partition_document
from ingestion.chunking import create_chunks_by_title
from ingestion.pipeline import summarise_chunks
from database.vector_store import get_vector_store


def run_pipeline(pdf_path):
    print("🚀 Starting RAG Pipeline")
    print("=" * 50)

    elements = partition_document(pdf_path)
    chunks = create_chunks_by_title(elements)
    docs = summarise_chunks(chunks)

    db = get_vector_store(docs)

    print("🎉 Pipeline completed successfully!")
    return db


if __name__ == "__main__":
    run_pipeline("data/docs/FOOD CHP1.pdf")