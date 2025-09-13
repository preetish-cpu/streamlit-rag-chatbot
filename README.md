# Streamlit RAG Chatbot
## Project Overview
This project is a PDF Chatbot with **Retrieval-Augmented Generation (RAG)**. It provides a user-friendly interface built with Streamlit that allows users to upload PDF documents and ask questions about their content. The chatbot leverages a RAG framework to ensure responses are accurate and grounded in the provided document's text.
## Features
- **PDF Document Querying**: Upload a PDF and interact with the content through a conversational interface.
- **Efficient Retrieval**: Uses **all-MiniLM-L6-v2** embeddings and a **FAISS** index to quickly find the most relevant text chunks within the PDF.
- **Powerful Language Model**: Integrates with the **Mistral 7B** large language model via the **OpenRouter API** to generate contextually relevant and accurate responses based on the retrieved information.
- **User-Friendly Interface**: The application is built with **Streamlit**, which turns data scripts into shareable web apps with an easy-to-use interface.
## Prerequisites
- Python
- An OpenRouter API Key
## Setup and Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/streamlit-rag-chatbot.git 
   cd streamlit-rag-chatbot
   ```
2. **Install the required libraries:**
    ```bash
    streamlit
    PyPDF2
    faiss-cpu
    sentence-transformers
    openai
    ```
    To install,run:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure your API key:**
   The application uses an environmental variable for the OpenRouter API key.Set the variable in your environment before running the app:
   ```bash
   export OPENROUTER_API_KEY="your_api_key_here"
   ```
## Usage
1. Run the application from your terminal:
   ```bash
   streamlit run RAG chatbot.py
   ```
2. The application will open in your web browser.
3. Upload a PDF file.
4. Once the embeddings are created, a text box will appear for you to ask questions about the document.
