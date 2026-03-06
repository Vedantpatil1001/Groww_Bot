import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables (like GOOGLE_API_KEY)
load_dotenv()

app = Flask(__name__)

# Global variable to hold our initialized RAG pipeline
rag_chain = None

def initialize_rag():
    global rag_chain
    print("Initializing RAG pipeline...")
    pdf_path = "data/groww_faq.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: Could not find the PDF file at {pdf_path}")
        return False

    try:
        # 1. Load the document
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # 2. Split into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        # 3. Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
        # 4. Store chunks in Vector Store (FAISS)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        
        # 5. Create a retriever to fetch top 3 most relevant chunks
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # 6. Initialize the Gemini LLM
        # Note: Using gemini-3-flash-preview exactly as requested.
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            temperature=0,
            max_retries=0
        )

        # 7. Create the prompt template
        system_prompt = (
            "You are a helpful and polite assistant for answering questions based on the Groww FAQ document. "
            "Use the following pieces of retrieved context to answer the question accurately. "
            "If the answer is not in the context provided, simply say that you don't know the answer, do not make it up. "
            "Keep your responses well-structured and easy to read.\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # 8. Combine into a retrieval chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        print("✅ RAG pipeline initialized successfully.")
        return True

    except Exception as e:
        print(f"❌ Failed to initialize RAG pipeline: {e}")
        return False


@app.route('/ask', methods=['POST'])
def ask():
    """
    API endpoint that receives a question and returns an answer 
    using the LangChain RAG pipeline.
    """
    global rag_chain
    if not rag_chain:
        return jsonify({"error": "Backend RAG system is not initialized."}), 500

    data = request.json
    question = data.get("question")
    
    if not question:
        return jsonify({"error": "No question provided in the request body."}), 400

    try:
        # Invoke the RAG chain
        response = rag_chain.invoke({"input": question})
        answer = response["answer"]
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Initialize the vector store and RAG chain just before the server starts
    success = initialize_rag()
    
    # Run the server on port 5000 if successful
    if success:
        print("🚀 Starting Flask server on http://127.0.0.1:5000")
        app.run(port=5000, debug=False)
    else:
        print("🛑 Server not started due to initialization failure.")
