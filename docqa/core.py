"""Core RAG pipeline for document question answering."""

import os
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains import (
        create_history_aware_retriever,
        create_retrieval_chain,
    )
    from langchain_classic.chains.combine_documents import (
        create_stuff_documents_chain,
    )
from langchain_text_splitters import RecursiveCharacterTextSplitter

_LOADER_MAP = {
    ".pdf": lambda p: PyPDFLoader(p, extract_images=False),
    ".txt": lambda p: TextLoader(p, encoding="utf-8"),
    ".csv": lambda p: CSVLoader(p),
    ".docx": lambda p: Docx2txtLoader(p),
    ".md": lambda p: TextLoader(p, encoding="utf-8"),
}


class DocumentQA:
    """Chat with any document using Retrieval-Augmented Generation.

    Examples:
        >>> qa = DocumentQA(openai_api_key="sk-...")
        >>> qa.index(["report.pdf", "notes.txt"])
        >>> answer = qa.ask("What are the key findings?")
        >>> print(answer)
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1500,
        chunk_overlap: int = 300,
        retrieval_k: int = 10,
    ):
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Provide openai_api_key or set the OPENAI_API_KEY env var."
            )
        os.environ["OPENAI_API_KEY"] = api_key

        self._llm = ChatOpenAI(model=model, temperature=0)
        self._embeddings = OpenAIEmbeddings(model=embedding_model)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._retrieval_k = retrieval_k

        self._vector_store: Optional[FAISS] = None
        self._rag_chain = None
        self._chat_history: list = []

    # -- public API --

    def index(self, file_paths: list[str | Path]) -> dict:
        """Index documents into the vector store.

        Args:
            file_paths: List of paths to PDF, TXT, CSV, DOCX, or MD files.

        Returns:
            Dict with ``documents`` and ``chunks`` counts.
        """
        docs = []
        for fp in file_paths:
            fp = Path(fp)
            loader_fn = _LOADER_MAP.get(fp.suffix.lower())
            if loader_fn is None:
                loader = UnstructuredFileLoader(
                    str(fp), mode="elements", strategy="fast"
                )
            else:
                loader = loader_fn(str(fp))
            docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            add_start_index=True,
        )
        chunks = splitter.split_documents(docs)

        self._vector_store = FAISS.from_documents(chunks, self._embeddings)
        self._build_rag_chain()

        return {"documents": len(docs), "chunks": len(chunks)}

    def ask(self, question: str) -> str:
        """Ask a question. Uses RAG if documents are indexed, else general chat.

        Args:
            question: The user's question.

        Returns:
            The assistant's answer as a string.
        """
        if self._rag_chain:
            response = self._rag_chain.invoke(
                {"input": question, "chat_history": self._chat_history}
            )
            answer = response["answer"]
        else:
            chain = self._general_chain()
            answer = chain.invoke(
                {"input": question, "chat_history": self._chat_history}
            )

        self._chat_history.append(HumanMessage(content=question))
        self._chat_history.append(AIMessage(content=answer))
        return answer

    def reset(self) -> None:
        """Clear chat history and indexed documents."""
        self._chat_history.clear()
        self._vector_store = None
        self._rag_chain = None

    @property
    def is_indexed(self) -> bool:
        return self._vector_store is not None

    @property
    def history(self) -> list[dict]:
        return [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for m in self._chat_history
        ]

    # -- internals --

    def _build_rag_chain(self) -> None:
        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Given a chat history and the latest user question which might "
                    "reference context in the chat history, formulate a standalone "
                    "question. Do NOT answer it, just reformulate if needed.",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        retriever = self._vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": self._retrieval_k}
        )
        history_retriever = create_history_aware_retriever(
            self._llm, retriever, contextualize_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an assistant for question-answering tasks. "
                    "Use the provided context to answer accurately. "
                    "If the context is insufficient, say so.\n\n{context}",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        qa_chain = create_stuff_documents_chain(self._llm, qa_prompt)
        self._rag_chain = create_retrieval_chain(history_retriever, qa_chain)

    def _general_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful AI assistant. Answer concisely."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        return prompt | self._llm | StrOutputParser()
