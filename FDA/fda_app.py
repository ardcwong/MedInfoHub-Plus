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
from annotated_text import annotated_text


st.markdown('<p style="font-size: 14px; color: red; text-align: center;"><strong>⚠️ PharmaPal is designed to supplement, not replace, professional medical and pharmaceutical advice. We strongly encourage consulting a healthcare professional before making any medical decision. ⚠️</strong></p>', unsafe_allow_html=True)


api_key = st.secrets['api_key']
openai.api_key = api_key
client = OpenAI(api_key=api_key)
# SKLLMConfig.set_openai_key(api_key)
# Constants

CHROMA_DATA_PATH = 'fda_drugs_v6'
COLLECTION_NAME = "fda_drugs_embeddings_v6"


if "client_chromadb" not in st.session_state:
    st.session_state.client_chromadb = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
client_chromadb = st.session_state.client_chromadb

if "embed_func" not in st.session_state:
    st.session_state.embed_func = embedding_functions.OpenAIEmbeddingFunction(api_key=openai.api_key, model_name="text-embedding-ada-002")
openai_ef = st.session_state.embed_func


if "collection" not in st.session_state:
    # Create or get the collection
    st.session_state.collection = client_chromadb.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"}
    )

collection = st.session_state.collection
    


def return_best_drugs(user_input, collection, n_results=5):  # UPDATED
    query_result = collection.query(query_texts=[user_input], n_results=n_results)
    
    if not query_result['ids'] or not query_result['ids'][0]:
        print("No drugs found matching the query.")
        return []  # No results found

    top_results = []
    
    for i in range(min(n_results, len(query_result['ids'][0]))):  # UPDATED
        result_id = query_result['ids'][0][i]
        result_metadata = query_result['metadatas'][0][i]
        result_document = query_result['documents'][0][i]
        
        drug_name = result_metadata.get('drug', 'Unknown Drug')
        drug_description = result_document
        
        print(f"Drug {i+1}:")
        print("---------------")
        print(f"Name: {drug_name}")
        print("\nDrug Description:")
        print("-----------------")
        print(drug_description)
        
        top_results.append((drug_name, drug_description, result_id))  # UPDATED
    
    return top_results  # CHANGED


def disable_openai(x):
    if x == "Yes":
        disable = 1
    else:
        disable = 0
    return disable

def extract_keywords(drug_document):
    x = "No"

    disable = disable_openai(x)
    if disable == 1:
        return []
    else:
        try:
            response = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[
                    {"role": "system", "content": "You are a medical assistant bot tasked to extract keywords from the retrieved drug information."},
                    {"role": "assistant", "content": f"This is the retrieved information about the drug: {drug_document}"},
                    {"role": "user", "content": "Extract the five most crucial keywords from the retrieved drug information. Extracted keywords must be listed in a comma-separated list."}
                ]
            )
            top_keywords = response.choices[0].message.content
            return [kw.strip() for kw in top_keywords.split(',')]

        except:
            return []

# Summary and usage guidelines function based on user input and profile
def generate_user_conversational_response(drug_name, drug_document, user_profile):  # UPDATED
    if user_profile == "Patient":
        user_tone = "Explain the information in layman terms, focusing on the essential points a patient should know."
    elif user_profile == "Healthcare Provider":
        user_tone = "Provide detailed and technical information suitable for a healthcare provider."
    else:
        return "Please input healthcare_provider or patient."

    # Combined prompts for generating both the summary and usage guidelines based on user tone
    combined_messages = [
        {"role": "system", "content": "You are a medical assistant bot that generates both a summary and usage guidelines based on retrieved drug information."},
        {"role": "user", "content": f"This is the retrieved information about the drug {drug_name}: {drug_document}"},
        {"role": "user", "content": user_tone + " Generate both a summary (paragraph) and usage guidelines (bullet form) without any unnecessary formatting."}
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
    # keywords = extract_keywords(drug_document)
    return summary, usage_guidelines, keywords

#-------------MAIN PROGRAM---------------#
## GENERAL INFO AND INSTRUCTIONS
st.subheader("Welcome to ⚕️PharmaPal!")

tab1, tab2= st.tabs(["About the App    ", "How to Use    "])
with tab1:
    col1, col2 = st.columns([1,1])
    col1.image('data/Pharma.png')
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

# with tab3:

st.divider()

st.markdown("<h1 style='text-align: center;'>⚕️PharmaPal</h1>", unsafe_allow_html=True)
a, b = st.columns([1,2])
aa, bb, cc = st.columns([1,1,1])   
# st.write(st.session_state.role)
# def keep_query(search):
#     if search:
#         query_text_keep = query_text
#         st.session_state.keep = query_text_keep
#         return st.session_state.keep 

query_text = a.text_input("Please enter a medical condition or drug name: ")
# Example usage
user_profile = st.session_state.role # patient or healthcare_provider
# on streamlit: user_profile = st.radio("I am a: ", ("patient", "healthcare_provider"))

if query_text:
    top_results = return_best_drugs(query_text, collection)
    # st.write(top_results)
    df = pd.DataFrame(top_results, columns=["Drug_Name", "Details", "ID"])
    # st.write(df)
    drug_names = df["Drug_Name"].tolist()
    choose = b.selectbox(
            f'Results Related to "***{query_text}***"',
            (drug_names), help = f'Any Info', index = None)
    st.session_state.choose = choose
    a.caption(f"Press to View Information for {st.session_state.choose}.")
    if a.button("View Information", use_container_width = True, type = "primary"):
        selected_drug_details = df[df["Drug_Name"] == choose]
        # st.write(selected_drug_details)
        keywords = extract_keywords(selected_drug_details["Details"])
        drug_name = selected_drug_details["Drug_Name"].tolist()
        drug_document = selected_drug_details["Details"].tolist()

        # st.write(keywords)
        summary, usage_guidelines, top_keywords = generate_user_conversational_response(drug_name, drug_document, user_profile) 
        # st.write(f"Summary:\n-----------------\n{summary}\n\nUsage Guidelines:\n-----------------\n{usage_guidelines}")

        with st.container():
            # st.write(drug_name)
            st.markdown(f"<h2 style='text-align: center;'><b><i>{st.session_state.choose}</i></h2>", unsafe_allow_html=True)
            column1, column2 = st.columns([1,1])
            column1.subheader("Summary")
            column1.caption('TOP KEYWORDS')
            if top_keywords:
                highlighted_keywords = ""
                for i, keyword in enumerate(top_keywords):
                    highlighted_keywords += f"<span style='background-color:#FFD3D3;padding: 5px; border-radius: 5px; margin-right: 5px;'>{keyword}</span>"

                column1.markdown(highlighted_keywords, unsafe_allow_html=True)

            else:
                highlighted_tkw = ""
                highlighted_tkw += f"<span style='background-color:#96BAC5;padding: 5px; border-radius: 5px; margin-right: 5px;'>{'Top Keywords is unavailable.'}</span>"
                column1.markdown(highlighted_tkw, unsafe_allow_html=True)
            
            if summary:

                column1.markdown(summary)

            else:
                highlighted_summ = ""
                highlighted_summ += f"<span style='background-color:#96BAC5;padding: 5px; border-radius: 5px; margin-right: 5px;'>{'Summarizer is unavailable.'}</span>"
                column1.markdown(highlighted_summ, unsafe_allow_html=True)

            if usage_guidelines:
                column1.subheader("Usage and Guidelines")
                column1.markdown(usage_guidelines)

            else:
                highlighted_summ = ""
                highlighted_summ += f"<span style='background-color:#96BAC5;padding: 5px; border-radius: 5px; margin-right: 5px;'>{'Usage and Guidelines is unavailable.'}</span>"
                column1.markdown(highlighted_summ, unsafe_allow_html=True)
    

# " ".join(drug_document)
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(summary+usage_guidelines)
            st.session_state['wordcloud'] = wordcloud

            # Display the word cloud
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            column2.subheader("Word Cloud")
            column2.pyplot(plt)
            with column2:
                def telemedicine():
                    st.subheader('Telemedicine and Specialty Doctors')
                
                    # Original text with website titles and URLs
                    text = """
                    For telemedicine consultations or to find the nearest specialty doctor near you, you may visit:
                
                    <b>NowServing</b>: https://nowserving.ph/<br><b>Konsulta MD</b>: https://konsulta.md/<br><b>SeriousMD</b>: https://seriousmd.com/healthcare-super-app-philippines
                    """
                    # st.link_button("Now Serving", "https://nowserving.ph")
                    # st.link_button("Konsulta MD", "https://konsulta.md/")
                    # st.link_button("SeriousMD", "https://seriousmd.com/healthcare-super-app-philippines")
                    # Display formatted text with st.markdown
                    st.markdown(text, unsafe_allow_html=True)
                telemedicine()
    else:
        if st.session_state.choose == []:
            st.error("Choose one from results to view")
