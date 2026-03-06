# Simple RAG Chatbot Project

This is a **beginner-friendly, minimal RAG-based Chatbot** using Python, Flask, Streamlit, and LangChain.
It answers questions based on a specific document (`data/groww_faq.pdf`) utilizing the Google Gemini API (`gemini-3-flash-preview`).

## Features
- **Backend API:** Built with Flask, serving LLM completion with LangChain.
- **Frontend UI:** Built with Streamlit, providing an interactive Chat interface.
- **RAG Pipeline:** Utilizes PyPDFLoader, FAISS vector database, Google AI Embeddings, and Gemini LLM model.
- **Simple Architecture:** Only 2 primary source files to keep it completely understandable for beginners.

## Directory Structure
```
VedantBot/
├── .env                  # Environment file (API Keys)
├── backend.py            # Flask API & RAG Pipeline setup
├── frontend.py           # Streamlit Web User Interface
├── requirements.txt      # Python dependencies dependencies
├── README.md             # Project documentation
└── data/
    └── groww_faq.pdf     # Source knowledge base for the RAG system
```

## Setup Instructions

1. **Activate the Virtual Environment:**
   *(If not initialized, the `env` is already configured in this workspace)*
   **Windows:**
   ```powershell
   venv\Scripts\activate
   ```
   **Mac/Linux:**
   ```bash
   source venv/bin/activate
   ```

2. **Install the Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Provide API Key:**
   - Open the `.env` file in the root directory.
   - Replace `your_gemini_api_key_here` with your actual Google Gemini API token. Ensure not to upload this file publicly.

## Running the Application Locally

The application is split between a separate frontend and a backend.

**Step 1: Start the Backend (Flask API & RAG Server)**
Open your first terminal and run:
```bash
python backend.py
```
*Wait until the terminal outputs:* `✅ RAG pipeline initialized successfully.` and `🚀 Starting Flask server on http://127.0.0.1:5000`

**Step 2: Start the Frontend (Streamlit App)**
Open a **second terminal window**, activate your environment, and run:
```bash
streamlit run frontend.py
```
This will automatically open your web interface in a browser window (`http://localhost:8501`). Wait for it to connect to your backend, and you can now ask your Chatbot questions!
