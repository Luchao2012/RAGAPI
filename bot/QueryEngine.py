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


from metadata_filter import MetadataSearchEngine
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterCondition
from utils import load_prompt, load_prompt_with_memory
from llama_index.core import QueryBundle
from llama_index.postprocessor.cohere_rerank import CohereRerank
cohere_api_key = "PgrTMGuhSZqjVX3PcgDFzd3VCTRQkcVyxXj6UOa3"

class BaseQueryEngine:
    def __init__(self, index, llm, similarity_top_k=20, use_rerank=True, rerank_top_n=10, text_qa_template_path="prompts/prompt_1.txt", filters=None):
        self.index = index
        self.similarity_top_k = similarity_top_k
        self.text_qa_template_path = text_qa_template_path  # Store the template path
        self.llm = llm
        self.filters = filters
        self.use_rerank = use_rerank
        if self.use_rerank:
            #self.reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=rerank_top_n)
            self.reranker = CohereRerank(api_key=cohere_api_key, top_n=rerank_top_n)

    def query(self, question, chat_history=None):
        print(f"retrieving relevant noes for question: {question}")
        retriever = self.index.as_retriever(
            similarity_top_k=self.similarity_top_k,
            filters=self.filters)

        retrieved_nodes = retriever.retrieve(question)
        print(f"retrieve nodes completed")

        if self.use_rerank:
            logging.info("Using rerank")
            query_bundle = QueryBundle(question)
            reranked_nodes = self.reranker.postprocess_nodes(retrieved_nodes, query_bundle)
        else:
            logging.info("Not using rerank")
            reranked_nodes = retrieved_nodes

        context_str = ""
        source_data = {}
        for i, node in enumerate(reranked_nodes):
            context_str += f"Content: {node.text}. chunk_id: 'chk-{i+1}', source_name: {node.metadata['file_name']}{node.metadata['page_label']}."
            source_data[f"chk-{i+1}"] = {
                "text": node.text,
                "file_name": node.metadata['file_name'],
                "page": node.metadata['page_label']
            }
        context_str = context_str.replace('\n', '')

        if chat_history:
            self.text_qa_template = load_prompt_with_memory(self.text_qa_template_path, memory=chat_history)
        else:
            self.text_qa_template = load_prompt(self.text_qa_template_path)

        fmt_qa_prompt = self.text_qa_template.format(
            context_str=context_str, query_str=question
        )
        logging.info(f"Formatted QA prompt: {fmt_qa_prompt}")

        response = self.llm.complete(fmt_qa_prompt)

        return response, source_data

class QueryEngineWithTfidfFilter(BaseQueryEngine):
    def __init__(self, 
                 index, 
                 llm,  
                 semantic_weight=0.7, 
                 filter_top_n=5,
                 similarity_top_k=10, 
                 use_rerank=True,
                 rerank_top_n=5,
                 text_qa_template_path="prompts/prompt_1.txt", 
                 file_path_df='metadata_embedding_db.pkl', 
                 embed_model=embed_model
                 ):
        
        # Initialize the BaseQueryEngine with the necessary parameters
        super().__init__(
            index=index, 
            similarity_top_k=similarity_top_k, 
            text_qa_template_path=text_qa_template_path,  # Correct the parameter name
            llm=llm,
            rerank_top_n=rerank_top_n,
            filters=None,  # Ensure filters is explicitly set if needed
            use_rerank=use_rerank  # Pass the use_rerank parameter
        )
        
        # Additional properties specific to this subclass
        self.metadata_search_engine = MetadataSearchEngine(
            df_path=file_path_df,  # Path to the metadata DataFrame
            embed_model=embed_model  # Embedding model
        )
        self.top_n = filter_top_n
        self.semantic_weight = semantic_weight

    def query(self, question, chat_history=None):
        # Filter metadata based on the input question
        metadata_search = self.metadata_search_engine.filter(
            question,
            top_n=self.top_n,
            tfidf_weight=1-self.semantic_weight,
        )
        print(f"Metadata filter results: {metadata_search}")
        logging.info(f"Metadata filter results: {metadata_search}")
        # Create MetadataFilters instance using the results from metadata search
        self.filters = MetadataFilters(
            filters=[MetadataFilter(key="ReportFileName", value=item) for item in metadata_search],
            condition=FilterCondition.OR,
        )

        # Call the query method of the base class to perform the query with the updated filters
        return super().query(question, chat_history)

 
class QueryEngineWithFilter(BaseQueryEngine):
    def __init__(self, 
                 index, 
                 llm, 
                 filter_top_n, 
                 bm25_weight, 
                 semantic_weight, 
                 similarity_top_k=5, 
                 text_qa_template_path="prompts/prompt_1.txt", 
                 file_path_df='metadata_embedding_db.pkl', 
                 file_path_bm25='bm25_database.pkl', 
                 embed_model=embed_model):
        
        # Initialize the BaseQueryEngine with the necessary parameters
        super().__init__(
            index=index, 
            similarity_top_k=similarity_top_k, 
            text_qa_template_path=text_qa_template_path,  # Correct the parameter name
            llm=llm,
            filters=None  # Ensure filters is explicitly set if needed
        )
        
        # Additional properties specific to this subclass
        self.metadata_search_engine = MetadataSearchEngine(
            file_path_df=file_path_df,  # Path to the metadata DataFrame
            file_path_bm25=file_path_bm25,  # Path to the BM25 database
            embed_model=embed_model  # Embedding model
        )
        self.top_n = filter_top_n
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight

    def query(self, question):
        # Filter metadata based on the input question
        metadata_search = self.metadata_search_engine.filter(
            question,
            top_n=self.top_n,
            bm25_weight=self.bm25_weight,
            semantic_weight=self.semantic_weight
        )
        print(f"Metadata filter results: {metadata_search}")
        logging.info(f"Metadata filter results: {metadata_search}")
        # Create MetadataFilters instance using the results from metadata search
        self.filters = MetadataFilters(
            filters=[MetadataFilter(key="ReportFileName", value=item) for item in metadata_search],
            condition=FilterCondition.OR,
        )

        # Call the query method of the base class to perform the query with the updated filters
        return super().query(question)