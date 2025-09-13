# Trigger rebuild for apt-packages


import streamlit as st
import fitz  # PyMuPDF
import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from paddleocr import PaddleOCR
import faiss
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()

# --- CONFIG ---
#PDF_PATH = r"D:\prj\faq contents pdf.pdf"
st.title("🤖 PDF Chatbot Assistant")
uploaded_pdf = st.file_uploader("Upload your PDF", type=["pdf"])
if uploaded_pdf is not None:
    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")

OUTPUT_TEXT_FILE = "hybrid_ocr_output.txt"

# --- Initialize PaddleOCR ---
print("🚀 Initializing PaddleOCR...")
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # You can set lang='en+ch' if needed
with open(OUTPUT_TEXT_FILE, "w", encoding="utf-8") as out_file:
    for page_num, page in enumerate(doc):
        out_file.write(f"\n\n=== Page {page_num + 1} ===\n")

        # --- Extract Normal Text ---
        out_file.write("\n--- Normal Text ---\n")
        normal_text = page.get_text("text").strip()
        if normal_text:
            out_file.write(normal_text + "\n")
        else:
            out_file.write("(No selectable text found)\n")

        # --- Extract Images + OCR ---  ✅ THIS MUST BE INSIDE
        out_file.write("\n--- OCR on Images (as Table if possible) ---\n")
        image_list = page.get_images(full=True)

        if not image_list:
            out_file.write("(No images found)\n")
        else:
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
                    image_np = np.array(image_pil)

                    # Run OCR
                    result = ocr.ocr(image_np, cls=True)

                    if result:
                        table_rows = []
                        for line in result:
                            if line:
                                row_text = []
                                for entry in line:
                                    if entry and len(entry) == 2:
                                        box, (text, confidence) = entry
                                        row_text.append(text.strip())
                                if row_text:
                                    table_rows.append(" | ".join(row_text))

                        if table_rows:
                            for row in table_rows:
                                out_file.write(row + "\n")
                        else:
                            out_file.write("(No structured rows found)\n")
                    else:
                        out_file.write("(No OCR result found in image)\n")

                except Exception as e:
                    out_file.write(f"(Error processing image {img_index + 1}: {e})\n")

print("\n✅ Hybrid text + OCR extraction complete.")
print(f"📁 Results saved to: {OUTPUT_TEXT_FILE}")

def preprocess_text(documents):
    processed_docs = []
    for doc in documents:
        cleaned_text = "\n".join([line.strip() for line in doc.strip().split("\n") if line.strip()])
        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header")])
        chunks = header_splitter.split_text(cleaned_text)
        
        for chunk in chunks:
            recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
            processed_docs.extend(recursive_splitter.split_text(chunk.page_content))
    return processed_docs

def create_faiss_index(processed_docs, embeddings):
    if not processed_docs:
        return None, []
    doc_embeddings = embeddings.embed_documents(processed_docs)
    embedding_matrix = np.array(doc_embeddings).astype('float32')
    dim = embedding_matrix.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(embedding_matrix)
    return faiss_index, processed_docs

def search_faiss_index(query, faiss_index, embeddings, indexed_docs, top_k=5, threshold = 1.4 ):
    if faiss_index is None or faiss_index.ntotal == 0:
        return ["No documents available in the FAISS index."]
    top_k = min(top_k, faiss_index.ntotal)
    query_embedding = embeddings.embed_query(query)
    distances, indices = faiss_index.search(np.array([query_embedding]).astype('float32'), k=top_k)
    valid_results = [
        indexed_docs[idx]
        for idx in indices[0]
        if 0 <= idx < len(indexed_docs) and distances[0][list(indices[0]).index(idx)] < threshold
    ]
    return valid_results if valid_results else ["Data not present in the provided documents."]

def get_answer_from_llm(query, results, llm):
    prompt = (
        "You are an intelligent assistant designed to extract accurate and relevant information from an official SES document.\n\n"
        "Your behavior depends on the type of user question:\n"
        "- If the question asks for steps, procedures, or how to do something, respond with a clear, numbered step-by-step format.\n"
        "- If the question is factual, definitional, or yes/no type, give a short and direct answer.\n"
        "- If an SAP T-code is mentioned or relevant, always include it in your response.\n"
        "- If the information is not present in the extracted content, say so honestly.\n"
        "- **Also examine tabular and OCR-style structured data carefully and include details from it if it helps answer the question.**\n\n"
        "- **If the question is about SES creation, start the answer with 'Step 1:' and explicitly mention: 'The T-code for SES Creation is ML81N.'**\n\n"
        "Be concise, avoid unnecessary elaboration, and match the response style to the question type.\n\n"
        f"Extracted Content:\n{results}\n\n"
        f"User Question: {query}\n\n"
        "Answer:"
    )
    return llm.predict(prompt)
if __name__ == "__main__":
    #os.environ["OPENAI_API_KEY"] = "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"
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    # Read OCR output from text file
    with open(OUTPUT_TEXT_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Preprocess OCR text into chunks
    documents = preprocess_text([raw_text])

    # Generate embeddings and build FAISS index
    embeddings = OpenAIEmbeddings()
    faiss_index, indexed_docs = create_faiss_index(documents, embeddings)

    # LLM and query
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    #while True:
    query = st.text_input("Ask your question here 👇")
    if query :
        results = search_faiss_index(query, faiss_index, embeddings, indexed_docs)
        answer = get_answer_from_llm(query, "\n\n".join(results), llm)
        
        st.subheader("Answer : ")
        st.write(answer)

