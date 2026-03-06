import streamlit as st
import requests

# 1. Page Config
st.set_page_config(page_title="Groww FAQ Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Groww FAQ Chatbot")
st.markdown("💬 Ask me any question based on the **Groww FAQ** document!")

# 2. Setup Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Render Custom Chat Messages from History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. Handle User Input
user_input = st.chat_input("E.g., How do I open a Groww account?")

if user_input:
    # Display UI
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Save into session state
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display Bot Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing FAQ and formulating an answer..."):
            try:
                # 5. Call Flask Backend Running on localhost:5000
                backend_url = "http://localhost:5000/ask"
                response = requests.post(
                    backend_url, 
                    json={"question": user_input},
                    timeout=90 # 90 second timeout incase LLM takes long
                )
                
                # Check status and parse responses
                if response.status_code == 200:
                    bot_answer = response.json().get("answer", "No answer could be generated.")
                else:
                    bot_answer = f"⚠️ Server Error: {response.json().get('error', 'Unknown Error')}"
            
            except requests.exceptions.ConnectionError:
                bot_answer = (
                    "🚫 Could not connect to the backend server! "
                    "Make sure the Flask application (`backend.py`) is running on port 5000."
                )
            except Exception as e:
                bot_answer = f"❌ An unexpected error occurred: {str(e)}"

        # Render response block
        st.markdown(bot_answer)
    
    # Save into chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_answer})
