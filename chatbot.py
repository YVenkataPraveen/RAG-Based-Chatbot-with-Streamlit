import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader, UnstructuredFileLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# Set page config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Sidebar for API key and instructions
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    st.sidebar.warning("Please enter your OpenAI API key to start chatting.")
    st.stop()

st.sidebar.markdown("""
- **General Chat**: Start chatting immediately with the AI.
- **RAG Chat**: Upload files (PDF, TXT, CSV, DOCX, etc.) and index them to query your data.
- **Note**: For PDFs, ensure Poppler is installed and in PATH. Alternatively, use TXT or DOCX files.
- **Troubleshooting**: If answers are "I don't know," check if files loaded correctly or try more specific questions.
""")

# Main title
st.title("RAG-Based Chatbot with File Uploads")

# File uploader
uploaded_files = st.file_uploader("Upload your files (optional)", accept_multiple_files=True, type=['pdf', 'txt', 'csv', 'docx', 'md', 'html', 'json'])

# Session state for vector store, chat history, and indexed status
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# Function to select appropriate loader based on file extension
def get_loader(file_path: str):
    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".pdf":
        return PyPDFLoader(file_path, extract_images=False)
    elif extension == ".txt":
        return TextLoader(file_path, encoding="utf-8")
    elif extension == ".csv":
        return CSVLoader(file_path)
    elif extension == ".docx":
        return Docx2txtLoader(file_path)
    else:
        return UnstructuredFileLoader(file_path, mode="elements", strategy="fast")

# Function to load and index documents
def index_documents(uploaded_files):
    if not uploaded_files:
        st.warning("No files uploaded.")
        return None

    # Create a temporary directory to save uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        # Load documents from directory
        loader = DirectoryLoader(
            temp_dir,
            glob="**/*",
            loader_cls=get_loader,
            show_progress=True
        )
        try:
            docs = loader.load()
            st.session_state.doc_count = len(docs)
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")
            st.info("If this is a Poppler-related error, ensure Poppler is installed and in PATH. Alternatively, try uploading non-PDF files.")
            return None

        if not docs:
            st.error("No documents loaded.")
            return None

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Increased for more context
            chunk_overlap=300,  # Increased overlap
            add_start_index=True
        )
        chunks = text_splitter.split_documents(docs)
        st.session_state.chunk_count = len(chunks)

        # Embed and create vector store
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vector_store = FAISS.from_documents(chunks, embeddings)
            return vector_store
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return None

# Button to index files
if st.button("Index Uploaded Files") and uploaded_files:
    with st.spinner("Indexing documents... This may take a while."):
        st.session_state.vector_store = index_documents(uploaded_files)
        if st.session_state.vector_store:
            st.session_state.indexed = True
            st.success(f"Documents indexed successfully! Loaded {st.session_state.doc_count} documents, split into {st.session_state.chunk_count} chunks.")
        else:
            st.error("Indexing failed. Check the error messages above.")

# Display indexing status
if st.session_state.indexed:
    st.info(f"Indexed {st.session_state.doc_count} documents, {st.session_state.chunk_count} chunks. You can now query your data.")

# Setup LLM and prompts
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# General chat prompt (for non-RAG mode)
general_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. Answer the user's question concisely and accurately. If you don't know the answer, say so."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# General chat chain
general_chat_chain = general_prompt | llm | StrOutputParser()

# RAG setup (only if indexed)
if st.session_state.indexed:
    # Contextualize query prompt for RAG
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just reformulate it if needed."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Retriever
    retriever = st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})  # Increased k

    # History-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # QA prompt for RAG
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the provided context to answer the question as accurately as possible. "
        "If the context doesn't contain enough information, provide a brief answer based on what is available or say you need more details."
        "\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # QA chain
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Full conversational RAG chain
    conversational_rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# Chat interface
st.subheader("Chat with your data" if st.session_state.indexed else "General Chat")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Chat input
if prompt := st.chat_input("Ask a question" + (" about your documents" if st.session_state.indexed else "")):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    
    with st.spinner("Thinking..."):
        try:
            if st.session_state.indexed:
                # Use RAG chain
                response = conversational_rag_chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })
                answer = response["answer"]
                # Debug: Show retrieved context
                if response.get("context"):
                    with st.expander("Retrieved Context (Debug)"):
                        st.write("\n".join([doc.page_content for doc in response["context"]]))
            else:
                # Use general chat chain
                response = general_chat_chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })
                answer = response
            
            with st.chat_message("assistant"):
                st.markdown(answer)
            
            st.session_state.chat_history.append(AIMessage(content=answer))
        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            with st.chat_message("assistant"):
                st.markdown(error_message)
            st.session_state.chat_history.append(AIMessage(content=error_message))

# Reset button
if st.button("Reset Chat History"):
    st.session_state.chat_history = []
    if st.session_state.indexed:
        st.session_state.indexed = False
        st.session_state.vector_store = None
        st.session_state.doc_count = 0
        st.session_state.chunk_count = 0
    st.experimental_rerun()
