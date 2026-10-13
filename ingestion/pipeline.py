import json
from langchain_core.documents import Document
from services.llm_service import generate_summary  # NEW
import time

def separate_content_types(chunk):
    content_data = {
        "text": chunk.text,
        "tables": [],
        "images": []
    }

    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            if element_type == "Table":
                table_html = getattr(element.metadata, "text_as_html", element.text)
                content_data["tables"].append(table_html)

            elif element_type == "Image":
                if hasattr(element.metadata, "image_base64"):
                    content_data["images"].append(element.metadata.image_base64)

    return content_data


def summarise_chunks(chunks):
    print("🧠 Processing chunks with summaries...")

    documents = []

    for chunk in chunks:
        content = separate_content_types(chunk)

        # 🔥 TEXT stays same
        text_part = content["text"]

        # 🔥 TABLE SUMMARY
        table_summary = ""
        if content["tables"]:
            table_summary = generate_summary(
                "Summarize this table for retrieval:\n" + content["tables"][0]
            )

        # 🔥 IMAGE SUMMARY
        image_summary = ""
        if content["images"]:
            image_summary = generate_summary(
                "Describe this image for retrieval",
                image=content["images"][0]
            )
        
        time.sleep(2)

        # 🔥 FINAL COMBINED CONTENT
        final_content = f"""
        TEXT:
        {text_part}

        TABLE SUMMARY:
        {table_summary}

        IMAGE SUMMARY:
        {image_summary}
        """

        doc = Document(
            page_content=final_content,
            metadata={
                "original": json.dumps(content)
            }
        )

        documents.append(doc)

    print(f"✅ Processed {len(documents)} chunks")
    return documents