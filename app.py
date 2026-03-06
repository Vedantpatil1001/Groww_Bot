import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

# Handle possible langchain version differences
try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

# Load environment variables (for local testing, Streamlit Cloud uses its own Secrets management)
load_dotenv()

# 1. Page Config
st.set_page_config(page_title="Groww FAQ Chatbot", page_icon="🤖", layout="centered")

# 2. Initialize RAG using st.cache_resource so it runs only once per app instance
@st.cache_resource
def initialize_rag():
    pdf_path = "data/groww_faq.pdf"
    
    if not os.path.exists(pdf_path):
        st.error(f"❌ Error: Could not find the PDF file at {pdf_path}")
        return None

    try:
        # Load and Split Document
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Create Vectorstore
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Setup LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            temperature=0,
            max_retries=0
        )

        # Prompt Template
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

        # Build Chains
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return rag_chain

    except Exception as e:
        st.error(f"❌ Failed to initialize RAG pipeline: {e}")
        return None

# Show header
st.title("🤖 Groww FAQ Chatbot")
st.markdown("💬 Ask me any question based on the **Groww FAQ** document!")

# Show a loading spinner during the one-time initialization
with st.spinner("Initializing the Chatbot Database (This happens once)..."):
    rag_chain = initialize_rag()

# 3. Setup Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Handle User Input
user_input = st.chat_input("E.g., How do I open a Groww account?")

if user_input:
    # Display User message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display Bot Response
    with st.chat_message("assistant"):
        if not rag_chain:
            bot_answer = "⚠️ RAG pipeline is not initialized properly. Please check the logs."
            st.markdown(bot_answer)
        else:
            with st.spinner("Analyzing FAQ and formulating an answer..."):
                try:
                    response = rag_chain.invoke({"input": user_input})
                    bot_answer = response["answer"]
                except Exception as e:
                    bot_answer = f"❌ An error occurred: {str(e)}\n\nPlease ensure your `GOOGLE_API_KEY` is correctly set in Streamlit Secrets."
            st.markdown(bot_answer)
    
    # Save Assistant message
    st.session_state.messages.append({"role": "assistant", "content": bot_answer})
