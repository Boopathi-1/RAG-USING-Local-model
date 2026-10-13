import os
import json
from dotenv import load_dotenv
from google import genai
import base64
from google.genai import types

import json
import base64
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_answer(query, docs):
    text_parts = []
    image_parts = []

    docs = docs[:3]

    for doc in docs:
        # ✅ USE EMBEDDED CONTENT (THIS FIXES YOUR ISSUE)
        text_parts.append(doc.page_content)

        # OPTIONAL: still include images if available
        original = json.loads(doc.metadata.get("original", "{}"))

        if original.get("images"):
            try:
                image_bytes = base64.b64decode(original["images"][0])

                image_parts.append(
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg"
                    )
                )
            except Exception as e:
                print("⚠️ Image decode failed:", e)

    context = "\n\n".join(text_parts)
    print("\n🔴 FINAL CONTEXT SENT TO GEMINI:\n")
    print(context)
    prompt = f"""
    Answer using ONLY the context.

    CONTEXT:
    {context}

    QUESTION:
    {query}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt] + image_parts
        )
        

        return response.text

    except Exception as e:
        print("❌ Gemini error:", e)
        return "Error generating answer"
    
def generate_summary(prompt, image=None):
    try:
        if image:
            image_bytes = base64.b64decode(image)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    prompt,
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg"
                    )
                ]
            )
        else:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )

        return response.text

    except Exception as e:
        print("❌ Summary error:", e)
        return ""