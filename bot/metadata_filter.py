# Import dependencies and initate model
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
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

class MetadataSearchEngine:
    def __init__(self, df_path='df_with_embeddings.pkl', embed_model=None):
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update([
            "field", "reservoir", "formation", "basin", "area", "unit", "report", 
            "study", "well", "wells", "fields", "reservoirs", "formations", 
            "basins", "areas", "units", "reports", "studies"
        ])
        self.stop_words = list(self.stop_words)  # Convert set to list
        self.embed_model = embed_model
        self.load_and_process_df(df_path)

    def custom_tokenizer(self, text):
        words = re.split(r'\W+', text.lower())
        return [word for word in words if word not in self.stop_words and word.isalnum()]

    def load_and_process_df(self, df_path):
        # Load DataFrame
        with open(df_path, 'rb') as f:
            self.df = pickle.load(f)

        # Initialize and fit the TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=self.custom_tokenizer, stop_words=self.stop_words)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['combined_text'])

    def combine_columns(self, row):
        combined_text = ' '.join(str(row[col]) for col in self.df.columns)
        return combined_text.lower()

    def preprocess(self, text):
        """ Tokenizes and removes stopwords from the input text. """
        return self.custom_tokenizer(text)

    def filter(self, query, top_n=3, tfidf_weight=0.3, embedding_weight=0.7):
        # Normalize weights
        embedding_weight = 1 - tfidf_weight 

        # Preprocess and tokenize query
        tokenized_query = self.preprocess(query)
        query_text = ' '.join(tokenized_query)
    
        # Transform the query using the TF-IDF vectorizer
        query_tfidf = self.tfidf_vectorizer.transform([query_text])

        # Additional debug to print terms in the query vector
        print("Terms in query vector:")
        print(self.tfidf_vectorizer.inverse_transform(query_tfidf))
        
        # Compute cosine similarities between the query and the documents for TF-IDF
        cosine_similarities_tfidf = np.dot(query_tfidf, self.tfidf_matrix.T).toarray().flatten()

        # Compute cosine similarities for embeddings
        query_embedding = self.embed_model.get_text_embedding(query)
        cosine_similarities_embedding = np.array([np.dot(query_embedding, emb) for emb in self.df['embeddings']])

        # Combine TF-IDF and embedding similarities
        combined_scores = (cosine_similarities_tfidf * tfidf_weight) + (cosine_similarities_embedding * embedding_weight)
        
        # Get top N indexes based on combined scores
        top_indexes = np.argsort(combined_scores)[::-1][:top_n]
        
        # Retrieve top N matching rows
        top_matches = self.df.iloc[top_indexes]
        
        # Calculate and print combined similarities for top N matches
        print("Top N Matches and their Combined Similarities:")
        for i, index in enumerate(top_indexes):
            print(f"Rank {i + 1}:")
            print(f"Document: {self.df.iloc[index]['combined_text']}")
            print(f"Combined Similarity: {combined_scores[index]}")
            print(f"TF-IDF Cosine Similarity: {cosine_similarities_tfidf[index]}")
            print(f"Embedding Cosine Similarity: {cosine_similarities_embedding[index]}")
            print("-" * 80)
        
        return top_matches['ReportFileName'].tolist()