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
        persist_dir="../user_db", 
        collection_name="rag_exp",
        use_llama_parse=False):

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
    similarity_top_k: int = 10
    #semantic_weight: float = 0.7
    collection_name: str = "rag_exp"
    persist_dir: str = "../user_db"
    use_rerank: bool = True
    rerank_top_n: int = 5

# Define a model for the individual source data entries
class SourceData(BaseModel):
    id: str
    text: str
    file_name: str
    page: int

# Define the main response model
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
    
    #df_with_embeddings_path = "df_with_embeddings.pkl"

    # preprocess the question and chat history
    # if len(query.chat_history) == 0:
    #     new_question = query.query
    #     chat_history = None
    # else:
    #     chat_history = query.chat_history[-6:] # Only use the last 3 chat history interactions
    #     chat_history = [(key, value) for item in chat_history for key, value in item.items()] # Flatten the list of dictionaries to tuples
    #     print(f"chat_history is {chat_history}")
    #     # Reformulate the question based on the chat history
    #     new_question = reformulate_question(chat_history=chat_history, question=query.query)

    from QueryEngine import BaseQueryEngine
    query_engine = BaseQueryEngine(
        index=index,
        llm=llm_model,
        similarity_top_k=query.similarity_top_k,
        text_qa_template_path="prompts/prompt_1.txt",  # Make sure this path is correct
        use_rerank=query.use_rerank,
        rerank_top_n=query.rerank_top_n,)

    # query_engine = QueryEngineWithTfidfFilter(
    #     index=index,
    #     llm=llm_model,
    #     semantic_weight=query.semantic_weight,  # You can adjust this value as needed
    #     filter_top_n=query.filter_top_n,
    #     similarity_top_k=query.similarity_top_k,
    #     text_qa_template_path="prompts/prompt_1.txt",  # Make sure this path is correct
    #     file_path_df=df_with_embeddings_path,
    #     embed_model=embed_model,  # Embedding model you are using,
    #     use_rerank=query.use_rerank,
    #     rerank_top_n=query.rerank_top_n
    # )

    #response, source_data = query_engine.query(new_question, chat_history=chat_history)
    response, source_data = query_engine.query(query.query)
    # Convert source data to SourceData objects
    source_metadata = [
        SourceData(id=key, text=value["text"], file_name=value["file_name"], page=int(value["page"]))
        for key, value in source_data.items()
    ]

    return QueryResponse(
        response_text=response.text,
        source_metadata=source_metadata
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)