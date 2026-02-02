import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --------------------------------------------------
# Paths (portable for Docker & HF Spaces)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FAISS_PATH = os.path.join(BASE_DIR, "vectorstore", "db_faiss")

# --------------------------------------------------
# Load Vector Store (cached)
# --------------------------------------------------
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def is_greeting(text):
    return text.strip().lower() in ["hi", "hello", "hey", "greetings"]

def is_ok(text):
    return text.strip().lower() == "ok"

def is_no(text):
    return text.strip().lower() == "no"

# --------------------------------------------------
# Streamlit App
# --------------------------------------------------
def main():
    st.title("👨‍⚕️ MediBot – Medical Assistant 🩺")
    st.markdown("Ask medical questions based on trusted medical documents.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_response_was_query" not in st.session_state:
        st.session_state.last_response_was_query = False

    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Ask a medical question or say hi!")

    if not user_prompt:
        return

    # User message
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    # -------------------------------
    # Greeting handling
    # -------------------------------
    if is_greeting(user_prompt):
        reply = "Hello! 👋 How can I assist you today?"
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )
        st.session_state.last_response_was_query = False
        return

    # -------------------------------
    # Follow-up handling
    # -------------------------------
    if is_ok(user_prompt) and st.session_state.last_response_was_query:
        reply = "Glad I could help 🙂"
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )
        st.session_state.last_response_was_query = False
        return

    if is_no(user_prompt) and st.session_state.last_response_was_query:
        reply = "Thank you for using MediBot. Take care! 👨‍⚕️"
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )
        st.session_state.last_response_was_query = False
        return

    # -------------------------------
    # RAG pipeline
    # -------------------------------
    try:
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
        if not GROQ_API_KEY:
            st.error("❌ GROQ_API_KEY not found in environment variables.")
            return

        vectorstore = get_vectorstore()

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.4,
            max_tokens=512,
            api_key=GROQ_API_KEY,
        )

        # Medical-safe prompt (from your previous version)
        RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a medical assistant.

Using ONLY the information provided in the context,
give a clear, accurate, and easy-to-understand answer
for a non-medical user.

If the context does not contain enough information, say:
"I don't have enough information to answer this fully."

Context:
{context}

Question:
{input}

Answer:
""")

        combine_docs_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=RAG_PROMPT
        )

        rag_chain = create_retrieval_chain(
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            combine_docs_chain=combine_docs_chain
        )

        response = rag_chain.invoke({"input": user_prompt})
        answer = response["answer"].strip()

        final_answer = f"{answer}\n\nCan I help with any other query?"

        with st.chat_message("assistant"):
            st.markdown(final_answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": final_answer}
        )
        st.session_state.last_response_was_query = True

    except Exception as e:
        st.error(f"❌ Error processing query: {e}")
        st.session_state.last_response_was_query = False


if __name__ == "__main__":
    main()
