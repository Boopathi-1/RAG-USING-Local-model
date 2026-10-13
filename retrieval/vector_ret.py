from database.vector_store import get_vector_store

def vector_retrieve(query, k=5, debug=False):
    db = get_vector_store()
    retriever = db.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)

    if debug:
        print("\n🔵 VECTOR RETRIEVAL")
        for i, doc in enumerate(docs):
            print(f"\n[V{i+1}]")
            print(doc.page_content[:300])

    return docs