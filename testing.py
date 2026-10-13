from retrieval.hybrid import hybrid_retrieve as retrieve
from services.llm_service import generate_answer
import json

def ask_question(query):
    print("\n🔍 Retrieving relevant documents...")
    docs = retrieve(query, debug=True)

    print(f"\n📄 Retrieved {len(docs)} chunks\n")

    for i, doc in enumerate(docs):
        print("=" * 80)
        print(f"CHUNK {i+1}")
        print("=" * 80)

        # 🔹 Summary (what embedding used)
        print("\n🔸 EMBEDDED CONTENT:")
        print(doc.page_content[:500])  # limit output

        # 🔹 Original content
        original = json.loads(doc.metadata.get("original", "{}"))

        print("\n🔹 ORIGINAL TEXT:")
        print(original.get("text", "")[:300])

        if original.get("tables"):
            print("\n🔹 TABLE FOUND:")
            print(original["tables"][0][:300])

        if original.get("images"):
            print("\n🔹 IMAGE FOUND (base64 preview):")
            print(original["images"][0][:100])

        print("\n")

    print("\n🤖 Generating answer with Gemini...\n")
    answer = generate_answer(query, docs)

    print("💬 Question:", query)
    print("\n✅ Answer:\n", answer)


ask_question("What is HACCP?")