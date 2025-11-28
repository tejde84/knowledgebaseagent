import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.schema import StrOutputParser
import docx2txt
import json
import re

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"
CONFIDENCE_THRESHOLD = 0.35

st.set_page_config(page_title="Knowledge Base Agent", layout="wide")
st.title("📚 Knowledge Base Agent")
st.caption("Ask questions about your HR/Operations documents. Answers are grounded with citations.")

# Sidebar
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload PDFs, DOCX, or TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    rebuild = st.button("Rebuild Index")

# Load documents
def load_docs(files):
    docs = []
    for f in files:
        fname = f.name.lower()
        if fname.endswith(".pdf"):
            loader = PyPDFLoader(f)
            docs.extend(loader.load())
        elif fname.endswith(".txt"):
            loader = TextLoader(f, encoding="utf-8")
            docs.extend(loader.load())
        elif fname.endswith(".docx"):
            text = docx2txt.process(f)
            docs.append(type("Doc", (), {"page_content": text, "metadata": {"source": f.name}}))
    return docs

# Build FAISS index
def build_index(files):
    docs = load_docs(files)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    vectordb = FAISS.from_documents(chunks, embeddings)
    st.session_state["vectordb"] = vectordb
    st.success(f"Index built with {len(chunks)} chunks.")

# Load index if exists
if "vectordb" not in st.session_state and uploaded_files:
    build_index(uploaded_files)

if rebuild and uploaded_files:
    build_index(uploaded_files)

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """You are a helpful HR/Operations assistant. Use the context to answer concisely and cite sources.

If unsure or context is missing, say you're not confident and suggest escalating to HR.

Return a JSON line at the end with:
- confidence: 0 to 1
- sources: list of source file names

Question: {question}

Context:
{context}

Answer:"""
)

def format_docs(docs):
    return "\n\n".join([f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs])

def rag_chain(vectordb):
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    chain = RunnableParallel({"context_docs": retriever, "question": RunnablePassthrough()}) | (
        lambda x: {"context": format_docs(x["context_docs"]), "question": x["question"]}
    ) | prompt | llm | StrOutputParser()
    return chain

# Chat UI
st.markdown("### Ask a question")
query = st.text_input("Example: What is the maternity leave policy?")
ask = st.button("Ask")

if ask and query.strip():
    vectordb = st.session_state.get("vectordb", None)
    if vectordb is None:
        st.error("No index found. Upload documents and rebuild.")
    else:
        chain = rag_chain(vectordb)
        with st.spinner("Thinking..."):
            result = chain.invoke({"question": query})

        # Parse JSON line
        conf = 0.2
        sources = []
        matches = re.findall(r"\{.*?\}", result.replace("\n", " "), flags=re.DOTALL)
        for m in reversed(matches):
            try:
                j = json.loads(m)
                conf = float(j.get("confidence", conf))
                sources = j.get("sources", sources)
                break
            except:
                continue

        st.write(result.split("{")[0].strip())
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{conf:.2f}")
        with col2:
            st.write("Sources:", ", ".join(sources) if sources else "N/A")

        if conf < CONFIDENCE_THRESHOLD:
            st.warning("Low confidence. Consider escalating to HR or checking the original document.")

        # Save history
        if "history" not in st.session_state:
            st.session_state["history"] = []
        st.session_state["history"].append(
            {"question": query, "answer": result, "confidence": conf, "sources": sources}
        )

# Chat history
st.markdown("---")
st.markdown("### Chat History")
for i, h in enumerate(reversed(st.session_state.get("history", [])[-10:]), 1):
    st.write(f"Q{i}: {h['question']}")
    st.write(f"A{i}: {h['answer'].split('{')[0].strip()}")
    st.caption(f"Confidence: {h['confidence']:.2f} | Sources: {', '.join(h['sources']) if h['sources'] else 'N/A'}")