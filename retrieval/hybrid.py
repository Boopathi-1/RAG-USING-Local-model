from retrieval.vector_ret import vector_retrieve
from retrieval.bm25_ret import BM25Retriever
from database.vector_store import get_vector_store

def reciprocal_rank_fusion(results_list, k=60, debug=False):
    scores = {}

    for source_idx, result in enumerate(results_list):
        for rank, doc in enumerate(result):
            doc_id = doc.page_content

            if doc_id not in scores:
                scores[doc_id] = 0

            scores[doc_id] += 1 / (k + rank + 1)

            if debug:
                print(f"Doc from source {source_idx} rank {rank} → score += {1/(k+rank+1):.4f}")

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    doc_map = {doc.page_content: doc for result in results_list for doc in result}

    final_docs = [doc_map[doc_id] for doc_id, _ in sorted_docs[:5]]

    if debug:
        print("\n🟣 RRF FINAL RANKING")
        for i, (doc_id, score) in enumerate(sorted_docs[:5]):
            print(f"\n[R{i+1}] Score: {score:.4f}")
            print(doc_id[:300])

    return final_docs
def hybrid_retrieve(query, debug=False):
    db = get_vector_store()

    # Vector
    vector_docs = vector_retrieve(query, debug=debug)

    # BM25
    docs = db.get()["documents"]
    from langchain_core.documents import Document
    all_docs = [Document(page_content=d) for d in docs]

    bm25 = BM25Retriever(all_docs)
    bm25_docs = bm25.retrieve(query, debug=debug)

    # RRF
    final_docs = reciprocal_rank_fusion(
        [vector_docs, bm25_docs],
        debug=debug
    )

    if debug:
        print("\n🟡 FINAL DOCS SENT TO LLM")
        for i, doc in enumerate(final_docs):
            print(f"\n[F{i+1}]")
            print(doc.page_content[:300])

    return final_docs