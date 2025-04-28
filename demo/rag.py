import os

from dotenv import load_dotenv
from llama_index.core.chat_engine.condense_plus_context import CondensePlusContextChatEngine
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.pathway import PathwayRetriever
from traceloop.sdk import Traceloop

from pathway.xpacks.llm.vector_store import VectorStoreClient

load_dotenv()


Traceloop.init(app_name=os.environ.get("APP_NAME", "PW - LlamaIndex (Streamlit)"))

DEFAULT_PATHWAY_HOST = "demo-document-indexing.pathway.stream"

PATHWAY_HOST = os.environ.get("PATHWAY_HOST", DEFAULT_PATHWAY_HOST)

PATHWAY_PORT = int(os.environ.get("PATHWAY_PORT", "80"))


def get_additional_headers():
    headers = {}
    key = os.environ.get("PATHWAY_API_KEY")
    if key is not None:
        headers = {"X-Pathway-API-Key": key}
    return headers


vector_client = VectorStoreClient(
    PATHWAY_HOST,
    PATHWAY_PORT,
    additional_headers=get_additional_headers(),
)


retriever = PathwayRetriever(host=PATHWAY_HOST, port=PATHWAY_PORT)
retriever.client = VectorStoreClient(
    host=PATHWAY_HOST, port=PATHWAY_PORT, additional_headers=get_additional_headers()
)

llm = Ollama(model="gemma3:12b-it-qat", 
             base_url= "http://host.docker.internal:11434", 
             context_window=128000,
             request_timeout=120.0)

query_engine = RetrieverQueryEngine.from_args(
    retriever, llm,
)

startup_introduction = "Please allow a few minutes for new documents to be indexed. The underlying engine handles live document change for you. Query away."
DEFAULT_MESSAGES = [
    ChatMessage(role=MessageRole.USER, content="How to use this?"),
    ChatMessage(role=MessageRole.ASSISTANT, content=startup_introduction),
]

chat_engine = CondensePlusContextChatEngine.from_defaults(
    retriever=retriever,
    system_prompt="""You are RAG AI that answers users questions based on provided sources.
    IF QUESTION IS NOT RELATED TO ANY OF THE CONTEXT DOCUMENTS, SAY IT'S NOT POSSIBLE TO ANSWER USING PHRASE `The looked-up documents do not provde information about...`""",
    verbose=True,
    chat_history=DEFAULT_MESSAGES,
    llm=llm,
)
