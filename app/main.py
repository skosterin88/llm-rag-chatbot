import os
import uuid
from fastapi import FastAPI, File, UploadFile, Form, Request, Response, Cookie
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tempfile import NamedTemporaryFile
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.memory import ConversationBufferMemory
from langgraph.graph import StateGraph, END

from typing import List, TypedDict

class RAGState(TypedDict):
    question: str
    context: str
    citations: List[str]
    confidence: float


CHROMA_PATH = "./chroma_db"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://llm-server:11434")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# In-memory session store for demo (use Redis/DB for production)
session_memories = {}

def load_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    all_splits = text_splitter.split_documents(documents)
    return all_splits

def get_embedding_function_hf(model_name="all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

def index_documents(chunks, embedding_function, persist_directory=CHROMA_PATH):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    return vectorstore

def create_langgraph_chain(
    vector_store,
    memory,
    llm_model_name="gemma3:1b",
    context_window=8192,
    confidence_threshold=0.5
):
    llm = ChatOllama(
        model=llm_model_name,
        temperature=0,
        num_ctx=context_window,
        base_url=OLLAMA_BASE_URL
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 3, 'include_score': True}
    )
    template = """
    You are a helpful assistant. Use the conversation history and the provided context to answer the question.
    If the context does not contain enough information to answer, say "I don't know."
    Cite the sources you used by their page numbers if possible.

    Conversation history:
    {history}

    Context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    def retrieve_node(state):
        question = state["question"]
        # docs_and_scores = retriever.invoke(question)
        docs_and_scores = vector_store.similarity_search_with_score(question, k=3)
        if not docs_and_scores:
            return {"question": question, "context": "", "citations": [], "confidence": 0.0}
        context_chunks = []
        citations = []
        scores = []
        for doc, score in docs_and_scores:
            context_chunks.append(doc.page_content)
            page = doc.metadata.get("page", "unknown")
            citations.append(f"Page {page}")
            scores.append(score)
        confidence = max(scores) if scores else 0.0
        context = "\n\n".join(context_chunks)
        return {
            "question": question,
            "context": context,
            "citations": citations,
            "confidence": confidence
        }

    def llm_node(state):
        question = state["question"]
        context = state["context"]
        citations = state.get("citations", [])
        confidence = state.get("confidence", 0.0)
        history = memory.buffer

        if not context or confidence < confidence_threshold:
            answer = "I don't know."
            memory.save_context({"input": question}, {"output": answer})
            return {
                "question": question,
                "answer": answer,
                "citations": [],
                "confidence": confidence
            }

        prompt_value = prompt.invoke({
            "context": context,
            "question": question,
            "history": history
        })
        llm_response = llm.invoke(prompt_value)
        answer = output_parser.invoke(llm_response)
        if citations:
            answer += f"\n\nCitations: {', '.join(citations)}"
        memory.save_context({"input": question}, {"output": answer})
        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "confidence": confidence
        }

    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("llm", llm_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "llm")
    workflow.add_edge("llm", END)
    graph = workflow.compile()
    
    return graph

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, session_id: Optional[str] = Cookie(None)):
    if not session_id:
        session_id = str(uuid.uuid4())
    memory = session_memories.get(session_id)
    history = []
    if memory:
        for msg in memory.chat_memory.messages:
            if msg.type == "human":
                history.append({"role": "user", "text": msg.content})
            else:
                history.append({"role": "assistant", "text": msg.content})
    response = templates.TemplateResponse("index.html", {"request": request, "results": None, "history": history})
    response.set_cookie(key="session_id", value=session_id)
    return response

@app.post("/ask", response_class=HTMLResponse)
async def ask_questions(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    questions: str = Form(...),
    session_id: Optional[str] = Cookie(None)
):
    if not session_id:
        session_id = str(uuid.uuid4())
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(return_messages=True)
    memory = session_memories[session_id]

    question_list = [q.strip() for q in questions.split("\n") if q.strip()]
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        docs = load_documents(tmp_path)
        chunks = split_documents(docs)
        embedding_function = get_embedding_function_hf()
        vector_store = index_documents(chunks, embedding_function)
        graph = create_langgraph_chain(vector_store, memory)
        answers = []
        for question in question_list:
            state = {"question": question}
            result = graph.invoke(state)
            answers.append({
                "question": question,
                "answer": result["answer"],
                "citations": result.get("citations", []),
                "confidence": result.get("confidence", 0.0)
            })
        history = []
        for msg in memory.chat_memory.messages:
            if msg.type == "human":
                history.append({"role": "user", "text": msg.content})
            else:
                history.append({"role": "assistant", "text": msg.content})
        html_response = templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "results": answers,
                "history": history
            }
        )
        html_response.set_cookie(key="session_id", value=session_id)
        return html_response
    finally:
        os.remove(tmp_path)