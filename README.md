# 🩺 MediBot – AI Medical Chatbot using RAG

MediBot is an AI-powered chatbot designed for the medical domain.  
It uses **Retrieval Augmented Generation (RAG)** to provide accurate responses from documents, ensuring domain-specific knowledge and reliability.

---

## 📌 Project Overview

The chatbot is built in **three phases**:

1. **Setup Memory for LLM (Vector Database)**
   - Load raw PDF(s).
   - Split documents into **chunks**.
   - Generate **vector embeddings** for each chunk.
   - Store embeddings in **FAISS** (vector database).

2. **Connect Memory with LLM**
   - Setup **Mistral LLM** using HuggingFace.
   - Connect LLM with FAISS for retrieval.
   - Build a **LangChain Retrieval Chain** to handle queries.

3. **Setup User Interface**
   - Build chatbot UI using **Streamlit**.
   - Load vector store into cache for fast retrieval.
   - Implement **Retrieval Augmented Generation (RAG)** pipeline for responses.

---

## 🛠️ Tools & Technologies

- **LangChain** → Framework for LLM applications.  
- **HuggingFace** → Hub for pre-trained AI/ML models.  
- **Mistral** → Chosen Large Language Model.  
- **FAISS** → Vector database for storing embeddings.  
- **Streamlit** → Chatbot web interface.  
- **Python** → Core programming language.  
- **VS Code** → IDE for development.

---

## ⚙️ Technical Architecture

1. **Input Document(s)** → Upload PDFs.  
2. **Chunking & Embeddings** → Text is chunked and converted into vector embeddings.  
3. **FAISS Vector Store** → Embeddings stored for semantic search.  
4. **RAG Pipeline** → User query is matched against FAISS, relevant chunks retrieved, and passed to LLM.  
5. **LLM (Mistral)** → Generates contextual and accurate response.  
6. **Streamlit UI** → Interactive chatbot interface.  

---

## 🚀 How It Works (Step by Step)

1. **Document Processing**
   - Upload medical PDFs.
   - Convert into smaller text chunks.
   - Create embeddings (vector representation).

2. **Knowledge Base Setup**
   - Store embeddings in **FAISS** for semantic retrieval.

3. **LLM Integration**
   - Load Mistral LLM from HuggingFace.
   - Connect LLM with FAISS to answer queries contextually.

4. **User Interaction**
   - Deploy chatbot using Streamlit.
   - Query → FAISS retrieves relevant context → LLM generates response.

---

## 📈 Future Improvements

- 🔑 Add **authentication** in UI for secure access.  
- 📂 Enable **self-upload** of documents for dynamic knowledge base updates.  
- 📚 Support **multiple document embeddings** simultaneously.  
- ✅ Add **unit testing** for RAG components.  
- ☁️ Optionally deploy on **cloud platforms** (AWS, GCP, Azure).  

---

## 📝 Summary

- ✅ Built a **modern AI chatbot** for document-based Q&A.  
- ✅ Developed using a **3-phased modular approach**.  
- ✅ Implemented **Streamlit, LangChain, HuggingFace, Mistral, FAISS**.  
- ✅ Demonstrated an end-to-end **RAG pipeline**.  
- 🔮 Scope for further enhancements like multi-doc support, authentication, and deployment.

---

## 📂 Project Structure (Example)

