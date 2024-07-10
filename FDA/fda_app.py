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



def return_best_drug(user_input, collection, n_results=1):
    query_result = collection.query(query_texts=[user_input], n_results=n_results)
    
    if not query_result['ids'] or not query_result['ids'][0]:
        print("No drug found matching the query.")
        return None, None  # No results found

    # Get the top result
    top_result_id = query_result['ids'][0][0]
    top_result_metadata = query_result['metadatas'][0][0]
    top_result_document = query_result['documents'][0][0]
    
    # Print query results in a readable and formatted manner
    print("Top Drug Found:")
    print("---------------")
    print(f"Name: {top_result_metadata.get('drug', 'Unknown Drug')}")
    print("\nDrug Description:")
    print("-----------------")
    print(top_result_document)
    # print("\nRecommendation:")
    # print("----------------")
    
    return top_result_metadata.get('drug', 'Unknown Drug'), top_result_document, top_result_id

# Extracting keywords function
def extract_keywords(drug_document):
    try:
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": "You are a medical assistant bot tasked to extract keywords from the retrieved drug information."},
                {"role": "assistant", "content": f"This is the retrieved information about the drug: {json.dumps(drug_document)}"},
                {"role": "user", "content": "Extract the five most crucial keywords from the retrieved drug information. Extracted keywords must be listed in a comma-separated list."}
            ]
        )
        top_keywords = response.choices[0].message.content
        return [kw.strip() for kw in top_keywords.split(',')]
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []

# Summary and usage guidelines function based on user input and profile
def generate_user_conversational_response(user_input, collection, user_profile):
    relevant_drug_name, relevant_drug_document, top_result_id = return_best_drug(user_input, collection)
    
    if not relevant_drug_name:
        return "I couldn't find any relevant drug based on your input."

    if user_profile == "Patient":
        user_tone = "Explain the information in layman terms, focusing on the essential points a patient should know."
    elif user_profile == "Healthcare Provider":
        user_tone = "Provide detailed and technical information suitable for a healthcare provider."
    else:
        return "Please input healthcare_provider or patient."

    # Combined prompts for generating both the summary and usage guidelines based on user tone
    combined_messages = [
        {"role": "system", "content": "You are a medical assistant bot that generates both a summary and usage guidelines based on retrieved drug information."},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": f"This is the retrieved information about the drug {relevant_drug_name}: {relevant_drug_document}"},
        {"role": "user", "content": user_tone + " Generate both a summary and usage guidelines without any unnecessary formatting."}
    ]
    
    combined_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=combined_messages,
        max_tokens=400
    )
    
    combined_text = combined_response.choices[0].message.content
    
    # Separate the clinical summary and usage guidelines based on markers
    split_text = combined_text.split("Usage Guidelines:")
    
    if len(split_text) != 2:
        return combined_text  # Asked GPT to split text and to return the combined text if splitting doesn't work as expected
    
    summary = split_text[0].strip()
    usage_guidelines = split_text[1].strip()

    # Remove the "Summary:" marker (if present) after splitting text
    summary = summary.replace("Summary:", "").strip()

    # Extract top five keywords from the relevant_drug_document
    keywords = extract_keywords(relevant_drug_document)

    formatted_output = f"\nSummary:\n-----------------\n{summary}\n\nUsage Guidelines:\n-----------------\n{usage_guidelines}\n\nKeywords:\n{', '.join(keywords)}"
    return formatted_output

#-------------MAIN PROGRAM---------------#
## GENERAL INFO AND INSTRUCTIONS
st.subheader("Welcome to ⚕️PharmaPal!")

tab1, tab2, tab3 = st.tabs(["About the App    ", "How to Use    ","Ask PharmaPal"])
with tab1:
    col1, col2 = st.columns([1,1])
    col1.image('data/art.png')
    col2.write("")
    col2.write("")
    col2.write("")
    content = """
<b style='color:#0C3974;'>HealthPlus</b> empowers you with reliable medical knowledge, making healthcare information accessible to all through the <b style='color:#0C3974;'>provision of accessible and easy-to-understand medical information</b>. Leveraging the power of the MedQuAD dataset and advanced AI, it <b style='color:#0C3974;'>enhances public health literacy and supports telemedicine consultations.</b> Whether you’re a patient managing a chronic condition, a caregiver needing clear explanations, a healthcare provider requiring quick and reliable information, or a health enthusiast looking for health tips, MedInfoHub is your go-to resource for trusted medical knowledge.
"""
    col2.markdown(content, unsafe_allow_html=True)
    col2.write("*The MedQuAD dataset aggregates content from reputable sources like the National Institutes of Health (NIH), National Library of Medicine (NLM), and other authoritative medical organizations.")
with tab2:
    col1, col2 = st.columns([1,1])
    col1.image('data/art.png')
    col2.write("")
    col2.write("")
    col2.write("")
    col2.title("Instructions:")
    content_inst = """
    (1) Enter a Keyword to Search<br>(2) Choose Keyword Search Method<br>(3) Choose Focus Area (Applicable for Exact Word Search Method<br>(4) Retrieve Information about Focus Area<br><br>Focus Area: A category or specific subject within a broader topic that helps refine and target the search results more effectively.
    """
    col2.markdown(content_inst, unsafe_allow_html=True)

with tab3:
    a, b, c = st.columns([1,1,1])
    
    
    st.write(st.session_state.role)
    
    query_text = st.text_input("Please enter a medical condition or drug name: ")
    # Example usage
    user_profile = st.session_state.role # patient or healthcare_provider
    # on streamlit: user_profile = st.radio("I am a: ", ("patient", "healthcare_provider"))
    if query_text:
        relevant_drug_name, relevant_drug_document, top_result_id = return_best_drug(query_text, collection)
        formatted_output = generate_user_conversational_response(query_text, collection, user_profile)
        st.write(formatted_output)
        st.write(top_result_id)
    st.write(collection.count())
    st.write(collection.get(ids=['1000']))



