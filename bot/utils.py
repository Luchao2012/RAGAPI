#!pip install python-dotenv
from llama_index.core import VectorStoreIndex
import os
from llama_parse import LlamaParse  
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

from llama_index.core import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core import load_index_from_storage
import os


def build_sentence_window_index(
    document, llm, embed_model="local:BAAI/bge-small-en-v1.5", save_dir="sentence_index"
):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            [document], service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )

    return sentence_index


def get_sentence_window_query_engine(
    sentence_index,
    similarity_top_k=6,
    rerank_top_n=2,
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine


from llama_index.core.node_parser import HierarchicalNodeParser

from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core import StorageContext
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine


def build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index",
    chunk_sizes=None,
):
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes, storage_context=storage_context, service_context=merging_context
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=merging_context,
        )
    return automerging_index


def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=12,
    rerank_top_n=2,
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )
    return auto_merging_engine


from llama_index.core import ChatPromptTemplate
import ast

def load_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
        # Using ast.literal_eval to safely evaluate the string into a Python object
        prompt_tuples = ast.literal_eval(data)
        prompt = ChatPromptTemplate.from_messages(prompt_tuples)
        logging.info(f"Loaded prompt from {file_path}")
        #logging.info(f"Prompt: {prompt}")
    return prompt

def load_prompt_with_memory(file_path, memory):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
        # Using ast.literal_eval to safely evaluate the string into a Python object
        prompt_tuples = ast.literal_eval(data)

        if len(memory) > 0:       
            # Insert chat history into the prompt at the desired location
            insert_index = None
            for i, item in enumerate(prompt_tuples):
                if item == ("user", "{context_str}"):
                    insert_index = i + 1
                    break

            if insert_index is not None:
                # Insert chat history items into the prompt
                for chat_item in reversed(memory):
                    prompt_tuples.insert(insert_index, chat_item)

        prompt = ChatPromptTemplate.from_messages(prompt_tuples)
        logging.info(f"Loaded prompt from {file_path}")
        logging.info(f"Prompt: {prompt}")
    return prompt

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def reformulate_question(chat_history, question):

    instruction_to_system = """
    Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
    """

    question_maker_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", instruction_to_system),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    fmt_qa_maker_prompt = question_maker_prompt.format(
        chat_history=chat_history, question=question
    )

    response = llm_gpt4.complete(fmt_qa_maker_prompt)
    return response.text
