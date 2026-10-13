from database.vector_store import get_vector_store
def retrieve(query):
    db = get_vector_store()  # load existing DB


    vector_retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = vector_retriever.invoke(query)

    return docs