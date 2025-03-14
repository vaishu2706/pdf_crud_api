import os
import io
import re
import uuid
from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.vectorstores import InMemoryVectorStore
import fitz  # PyMuPDF
from PIL import Image

app = Flask(__name__)

# --- Configure your Gemini AI Key ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyC3c3m_Ls0n5BQQXrMeh7JR9tbaAMVbwEs"

# --- Set up Gemini Embeddings and LLM ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


def extract_images_from_pdf(pdf_path):
    """
    Extract images from PDF using PyMuPDF (fitz) and return a list of PIL Images.
    """
    images = []
    doc = fitz.open(pdf_path)

    for page_index, page in enumerate(doc):
        # Get all images on this page
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Convert image bytes to a PIL Image
            image_pil = Image.open(io.BytesIO(image_bytes))
            images.append(image_pil)

    return images


@app.route("/upload", methods=['POST'])
def upload_file():
    """
    POST endpoint that:
      1. Receives JSON with { file_path, question }.
      2. Extracts text from PDF and chunks it.
      3. Builds an in-memory vector store.
      4. Finds relevant chunks for the question.
      5. Uses Gemini AI to answer the question based on those chunks.
      6. Extracts images from the PDF (via PyMuPDF).
      7. Returns the answer, related text, and image count.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON payload received"}), 400

        file_path = data.get("file_path")
        question = data.get("question")

        if not file_path:
            return jsonify({"error": "No file path provided"}), 400

        if not os.path.exists(file_path):
            return jsonify({"error": f"File not found: {file_path}"}), 404

        # --- 1) Extract Text from PDF ---
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # --- 2) Chunk the Text ---
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)

        # Assign a unique document_id to each chunk
        for doc in all_splits:
            doc.metadata["document_id"] = str(uuid.uuid4())

        # --- 3) Create an In-Memory Vector Store ---
        vector_store = InMemoryVectorStore(embedding=embeddings)
        vector_store.add_documents(documents=all_splits)

        # --- 4) Search for Relevant Chunks ---
        results = vector_store.similarity_search(question, k=3)

        if not results:
            return jsonify({"error": "No relevant content found"}), 404

        # Combine the top chunks into a single string
        combined_text = " ".join([res.page_content for res in results])

        # --- 5) Use Gemini AI to Get an Answer ---
        # We'll instruct Gemini to provide the answer from the PDF text
        response = llm.invoke(
            f"Read the following text and answer the question accurately. "
            f"Do not summarize; provide the exact content from the PDF.\n\n"
            f"Text: '{combined_text}'\n"
            f"Question: {question}"
        )
        response_text = response.content.strip()

        # Clean the answer text
        cleaned_answer = re.sub(r'[*\-]', '', response_text)
        cleaned_answer = re.sub(r'\s+', ' ', cleaned_answer).strip()

        # Collect related text
        related_texts = []
        for res in results:
            clean_chunk_text = re.sub(r'\s+', ' ', res.page_content).strip()
            related_texts.append({
                "document_id": res.metadata.get("document_id"),
                "chunk_text": clean_chunk_text
            })

        # --- 6) Extract Images from the PDF using PyMuPDF ---
        images = extract_images_from_pdf(file_path)
        image_count = len(images)

        # --- 7) Return JSON Response ---
        return jsonify({
            "question": question,
            "answer": cleaned_answer,
            "relatedtext": related_texts,
            "image_count": image_count,
            "message": "Images extracted successfully" if image_count > 0 else "No images found"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
