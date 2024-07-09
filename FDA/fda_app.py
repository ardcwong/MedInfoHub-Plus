import streamlit as st

__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from skllm.config import SKLLMConfig
from skllm.models.gpt.text2text.summarization import GPTSummarizer
# from sentence_transformers import SentenceTransformer, util
import openai
from openai import OpenAI
from wordcloud import WordCloud
import subprocess
import time
import numpy as np
import ast #built in
import chromadb
from chromadb.utils import embedding_functions


api_key = st.secrets['api_key']
openai.api_key = api_key
client = OpenAI(api_key=api_key)
# SKLLMConfig.set_openai_key(api_key)
# Constants
CHROMA_DATA_PATH = 'fda_drugs'
COLLECTION_NAME = "fda_drugs_embeddings"

# Initialize ChromaDB client
client_chromadb = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai.api_key, model_name="text-embedding-ada-002")

# Create or get the collection
collection = client_chromadb.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"}
)

st.write(len(collection))
