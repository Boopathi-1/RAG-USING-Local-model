from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.corpus = [doc.page_content for doc in documents]
        self.tokenized_corpus = [doc.page_content.split() for doc in documents]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query, k=5, debug=False):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = [self.documents[i] for i in ranked_indices[:k]]

        if debug:
            print("\n🟢 BM25 RETRIEVAL")
            for i, doc in enumerate(results):
                print(f"\n[B{i+1}]")
                print(doc.page_content[:300])

        return results