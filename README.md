# 	:lab_coat:MediBot:health_worker:Private Medical Assistant Using 💻 LLM & Memory
* ### *This project builds a memory-aware medical chatbot using Hugging Face models, LangChain, and FAISS. It enables users to query medical documents through a conversational interface, powered by retrieval-augmented generation (RAG).*
* ### *It loads medical PDFs, processes the text into embeddings using HuggingFace models, and stores them in a FAISS vector database. When a user asks a question, it retrieves relevant information and uses the LLaMA-3.1 language model to generate clear, reliable, and easy-to-understand medical answers.*

![AI-healthcare-Den_Vitruk_-alamy](https://github.com/user-attachments/assets/eb63c0ac-a5f9-4223-9b83-ea4e2d152b14)

## Techniques Used :arrow_right:

* __Document Loading__ with DirectoryLoader and PyPDFLoader.

* __Text Chunking__ using Recursive Character Text Splitter for semantic coherence.

* __Embedding__ using sentence-transformers/all-MiniLM-L6-v2 via HuggingFace.

* __Vector Store__ implementation using FAISS.

* __Prompt Engineering__ with Prompt Template.

* __LLM Inference__ via HuggingFaceEndpoint with controlled generation (top_k, top_p, temperature).

* __Session Memory__ with st.session_state in Streamlit for chat history.

* __Streamlit Chat UI__ with st.chat_input() and st.chat_message().

## Project Structure :arrow_right:

* __create_memory_for_llm.py__: Loads PDFs, splits content, generates embeddings, and stores in FAISS.

* __connect_memory_with_llm.py__ : CLI-based interface for querying the vector DB using LLaMA-3.1.Standalone script to test retrieval and generation in one go.

* __vectorstore__: Stores the serialized FAISS index.

*__App.py__: Full-featured Streamlit chatbot with conversational memory.

## Live demo available here :- https://huggingface.co/spaces/cprathamesh1997/Medical_Issues_Chatbot
