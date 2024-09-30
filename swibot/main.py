from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from pydantic import ValidationError
from typing import Dict, Any
from typing import List, Optional, Tuple
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Set up CORS middleware options
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

from llama_index.core import VectorStoreIndex
import os 
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
# index the documents and save to chromadb
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, ServiceContext
import openai
from getpass import getpass
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv('.env')
openai.api_key = os.environ['OPENAI_API_KEY']

import logging
# Add logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

llm_gpt35 = OpenAI(
    model="gpt-35-turbo",
)

llm_gpt4 = OpenAI(
    model="gpt-4",
)

# You need to deploy your own embedding model as well as your own chat completion model
embed_model = OpenAIEmbedding(
    model="text-embedding-ada-002",
)

from llama_index.core import Settings

Settings.llm = llm_gpt4
Settings.embed_model = embed_model

def get_llm_by_id(llm_id):
    if llm_id == "llm_gpt35":
        return llm_gpt35
    elif llm_id == "llm_gpt4":
        return llm_gpt4
    else:
        raise ValueError(f"Unsupported llm identifier: {llm_id}")

def load_chroma_index(
        persist_dir="swi_test_db", 
        collection_name="rag_swi_CA1039",
        use_llama_parse=True):

    logging.info(f"Loading chroma index {persist_dir}, collection name {collection_name}...")
    
    if use_llama_parse:
        node_parser = MarkdownNodeParser()
    else:
        node_parser = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=100)
    
    # Initialize client, setting path to save data
    db = chromadb.PersistentClient(path=persist_dir)

    # Create collection
    chroma_collection = db.get_or_create_collection(collection_name)

    # Assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create index
    chroma_index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context, node_parser=node_parser
    )
    
    return chroma_index

class QueryInput(BaseModel):
    query: str = "what is surfactants?"
    #chat_history: List[Dict[str, str]] = Field(default_factory=list)
    llm: str = "llm_gpt4"
    #filter_top_n: int = 5
    similarity_top_k: int = 5
    #semantic_weight: float = 0.7
    collection_name: str = "rag_swi_CA1039"
    persist_dir: str = "swi_test_db"
    use_rerank: bool = True
    rerank_top_n: int = 5

class SourceData(BaseModel):
    id: str
    text: str
    file_name: str
    page: int

class QueryResponse(BaseModel):
    response_text: str = Field(..., example="I do not have the relevant answer. Please refer to the searched documents for more information.")
    source_metadata: List[SourceData] = Field(..., example=[
        {
            "id": "chk-1",
            "text": "The outer shelf occupied the area between the rim and the platform slope and was deposited on slopes of 10-15°.",
            "file_name": "DAKS Kashagan N2017 Report.pdf",
            "page": 8
        }
    ])

@app.post("/qa/", response_model=QueryResponse, summary="Query the document index", response_model_exclude_unset=True)
async def qa_endpoint(query: QueryInput):
    if query.query.strip() == "":
        raise HTTPException(status_code=400, detail="The query must not be empty.")
    
    llm_model = get_llm_by_id(query.llm)  # Resolve the model

    # Update Settings configuration directly
    Settings.llm = llm_model
    Settings.embed_model = embed_model
    
    index = load_chroma_index(
        persist_dir=query.persist_dir,
        collection_name=query.collection_name)
    
    # create a query engine and query
    query_engine = index.as_query_engine(
        similarity_top_k=5, verbose=True
    )

    response = query_engine.query(query.query)

    # Return the response

    # Convert source nodes to SourceData objects
    source_metadata = [
        SourceData(
            id=node.id_, 
            text=node.get_content(), 
            file_name=node.metadata.get("file_name"), 
            page=int(node.metadata.get("page_number"))
        ) 
        for node in response.source_nodes
    ]

    return QueryResponse(
        response_text=response.response,
        source_metadata=source_metadata
    )