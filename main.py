import os
import streamlit as st
from dotenv import load_dotenv

from euriai.langchain import create_chat_model
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==============================
# Load API key from .env
# ==============================
load_dotenv()
EURI_API_KEY = os.getenv("EURI_API_KEY")

DB_FAISS_PATH = "vectorstore/db_faiss"

# ==============================
# Cache vectorstore for performance
# ==============================
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


# ==============================
# Custom Prompt
# ==============================
CUSTOM_PROMPT_TEMPLATE = """
You are GrabAI MediBot, a medical assistant chatbot.
Use the pieces of information provided in the context to answer user's question.
If you don’t know the answer, just say that you don’t know. 
Don’t try to make up an answer. Only use the given context.

Context: {context}
Question: {question}

Start the answer directly. Be precise and professional.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])


# ==============================
# Load LLM (EuriAI)
# ==============================
def load_llm():
    return create_chat_model(
        api_key=EURI_API_KEY,
        model="gpt-4.1-nano",
        temperature=0.5
    )


# ==============================
# Streamlit Custom Styling
# ==============================
st.set_page_config(page_title="GrabAI MediBot", page_icon="🩺", layout="centered")

st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #0a0a0f, #14142b);
        color: #e4e4e4;
    }

    /* Title */
    .stApp h1 {
        text-align: center;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .stApp p {
        text-align: center;
        color: #aaa;
        margin-bottom: 20px;
    }

    /* Chat bubbles */
    .stChatMessage {
        border-radius: 16px;
        padding: 14px 18px;
        margin: 10px 0;
        box-shadow: 0 4px 18px rgba(0,0,0,0.5);
    }

    /* User (green) */
    .stChatMessage[data-testid="stChatMessage-user"] {
        background: rgba(0, 180, 90, 0.25);
        border-left: 4px solid #00ff88;
        color: #ffffff;
    }

    /* Assistant (blue) */
    .stChatMessage[data-testid="stChatMessage-assistant"] {
        background: rgba(0, 120, 255, 0.25);
        border-left: 4px solid #00c6ff;
        color: #e0e0e0;
    }

    /* Input box */
    .stChatInput input {
        background-color: #1c1c2d !important;
        color: #ffffff !important;
        border-radius: 12px;
        border: 1px solid #444;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 30px;
        color: #777;
        font-size: 0.9em;
    }

    /* Sidebar */
    .sidebar-title {
        font-size: 1.1em;
        font-weight: bold;
        color: #00c6ff;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==============================
# Streamlit App
# ==============================
def main():
    st.title("🩺 GrabAI MediBot")
    st.caption("💡 Your AI-powered medical assistant")

    # Sidebar for file upload & actions
    st.sidebar.markdown("<div class='sidebar-title'>📂 Upload Medical Documents</div>", unsafe_allow_html=True)
    uploaded_files = st.sidebar.file_uploader("Upload PDF/Text/Docs", type=["pdf", "txt", "docx"], accept_multiple_files=True)

    if uploaded_files:
        st.sidebar.success(f"{len(uploaded_files)} file(s) uploaded successfully ✅")

    # Buttons
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.sidebar.info("Chat cleared!")

    if st.sidebar.button("📋 Show Uploaded Docs"):
        if uploaded_files:
            st.sidebar.write([f.name for f in uploaded_files])
        else:
            st.sidebar.warning("No documents uploaded yet.")

    # Init chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        st.chat_message(message["role"], avatar=avatar).markdown(message["content"])

    # User input
    prompt = st.chat_input("Ask your medical question...")

    if prompt:
        st.chat_message("user", avatar="👤").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("⚠️ Failed to load the vector store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=False,
                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
            )

            response = qa_chain.invoke({"query": prompt})
            result = response["result"]

            st.chat_message("assistant", avatar="🤖").markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    # Footer
    st.markdown("<div class='footer'>⚡ GrabAI MediBot | Produced by Amit Maji</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
