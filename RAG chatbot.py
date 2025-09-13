import streamlit as st
from PyPDF2 import PdfReader
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

# --- Load API key ---
api_key = os.getenv("OPENROUTER_API_KEY")

st.title("📄 PDF Chatbot with RAG")

if not api_key:
    st.error("⚠ OPENROUTER_API_KEY is not set. Please set it in your environment.")
else:
    # Init OpenRouter client
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    # Embedding model
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # --- Upload PDF ---
    pdf_file = st.file_uploader("Upload your PDF", type="pdf")

    if pdf_file:
        try:
            reader = PdfReader(pdf_file)
            text = "".join([page.extract_text() or "" for page in reader.pages])
        except Exception as e:
            st.error(f"❌ PDF read error: {e}")
            st.stop()

        # --- Chunk text ---
        def chunk_text(text, size=500, overlap=50):
            chunks = []
            for i in range(0, len(text), size - overlap):
                chunks.append(text[i:i+size])
            return chunks

        chunks = chunk_text(text)
        st.write(f"📑 Extracted {len(chunks)} chunks from PDF")

        # --- Embed & FAISS ---
        if "index" not in st.session_state or st.session_state.get("last_file") != pdf_file.name:
            with st.spinner("🔍 Creating embeddings..."):
                embeddings = embed_model.encode(chunks, normalize_embeddings=True)
                dim = embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)  # cosine similarity
                index.add(np.array(embeddings, dtype="float32"))

                st.session_state.index = index
                st.session_state.chunks = chunks
                st.session_state.last_file = pdf_file.name

        # --- Query ---
        query = st.text_input("Ask a question about the PDF:")
        if query:
            with st.spinner("🤖 Thinking..."):
                query_emb = embed_model.encode([query], normalize_embeddings=True)
                D, I = st.session_state.index.search(np.array(query_emb, dtype="float32"), k=3)
                context = " ".join([st.session_state.chunks[i] for i in I[0]])

                try:
                    response = client.chat.completions.create(
                        model="mistralai/mistral-7b-instruct",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": f"Answer the question using only this context:\n{context}\n\nQuestion: {query}"}
                        ]
                    )
                    st.write("💡 Answer:", response.choices[0].message.content)
                except Exception as e:
                    st.error(f"❌ API error: {e}")