import os
from dotenv import load_dotenv
from euriai.langchain import create_chat_model
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ==============================
# Step 1: Load API Key from .env
# ==============================
# .env file should have:
# EURI_API_KEY=your_api_key_here

load_dotenv()
EURI_API_KEY = os.getenv("EURI_API_KEY")

EURI_MODEL = "gpt-4.1-nano"
TEMPERATURE = 0.5

def load_llm():
    llm = create_chat_model(
        api_key=EURI_API_KEY,
        model=EURI_MODEL,
        temperature=TEMPERATURE
    )
    return llm


# ==============================
# Step 2: Custom Prompt
# ==============================

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don’t know the answer, just say that you don’t know. 
Don’t try to make up an answer. Only use the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt


# ==============================
# Step 3: Load FAISS Database
# ==============================

DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


# ==============================
# Step 4: Create QA Chain
# ==============================

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)


# ==============================
# Step 5: Run User Query
# ==============================

# if __name__ == "__main__":
#     user_query = input("Write Query Here: ")
#     response = qa_chain.invoke({'query': user_query})

#     print("RESULT: ", response["result"])
#     print("\nSOURCE DOCUMENTS:")
#     for doc in response["source_documents"]:
#         print(" - ", doc.metadata.get("source", "Unknown"), ":", doc.page_content[:200], "...")



user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})

print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])