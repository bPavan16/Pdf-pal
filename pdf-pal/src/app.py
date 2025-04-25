import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

import os
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

# Load environment variables and configure API
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return ""


def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, chunk_overlap=1000
        )
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error chunking text: {str(e)}")
        return []


def get_vector_store(text_chunks):
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("Google API key not found. Please check your .env file.")
            return None

        # Explicitly pass API key to embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=api_key
        )
        vector_store = FAISS.from_texts(text_chunks, embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None


def get_conversational_chain():
    prompt_template = """
    You are a helpful assistant. Answer the question as detailed as possible based on the context provided. If the answer is not in the context, say "I don't know", don't provide any wrong answers.
    Context: {context}
    Question: {question}
    
    Answer:
    """

    try:
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.4)
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None


def main():
    st.set_page_config(page_title="PDF Chat", layout="wide")

    # Initialize session state
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

    # Main layout
    st.title("Chat with Multiple PDFs")
    st.write("Upload your PDFs and ask questions based on their content.")

    # Sidebar for uploads and history
    with st.sidebar:
        st.title("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Upload your PDF files here", type=["pdf"], accept_multiple_files=True
        )

        # Only show start button if files are uploaded
        if uploaded_files:
            start_button = st.button("Process PDFs")

        # Display Q&A History in a collapsible section
        with st.expander("Q&A History", expanded=False):
            for i, (q, a) in enumerate(st.session_state.qa_history):
                st.markdown(f"**Q{i+1}:** {q}")
                st.markdown(f"**A{i+1}:** {a}")
                st.markdown("---")

    # Process PDFs when button is clicked
    if "start_button" in locals() and start_button and uploaded_files:
        # Check if the same files are already processed
        current_files = [f.name for f in uploaded_files]
        if current_files != st.session_state.processed_files:
            with st.spinner("Processing PDFs..."):
                pdf_text = get_pdf_text(uploaded_files)
                if pdf_text:
                    text_chunks = get_text_chunks(pdf_text)
                    if text_chunks:
                        vector_store = get_vector_store(text_chunks)
                        if vector_store:
                            st.session_state.vector_store = vector_store
                            st.session_state.processed_files = current_files
                            st.success(
                                "PDFs processed successfully! You can now ask questions."
                            )
        else:
            st.info("These PDFs are already processed.")

    # Display the chat interface if vector store is ready
    if "vector_store" in st.session_state:
        question = st.text_input("Ask a question about your PDFs:")
        if question:
            chain = get_conversational_chain()
            if chain:
                with st.spinner("Fetching answer..."):
                    try:
                        docs = st.session_state.vector_store.similarity_search(question)
                        answer = chain.run(input_documents=docs, question=question)

                        # Add to history and display the answer
                        st.session_state.qa_history.append((question, answer))

                        # Display in a nice format
                        st.markdown("### Answer")
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")


if __name__ == "__main__":
    main()
