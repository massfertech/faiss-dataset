import streamlit as st
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import nltk
import re
from nltk.corpus import stopwords
import os

st.set_page_config(layout="wide")
nltk.download('stopwords')

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device="cpu", trust_remote_code=True)

@st.cache_resource
def load_faiss_index():
    return faiss.read_index("data/faiss_index (4).bin")

@st.cache_data
def load_embeddings():
    return np.load("data/embeddings (4).npy")

@st.cache_data
def load_dataframe():
    parts = [pd.read_parquet(f"data/final_df_21_02_part_{i}.parquet") for i in range(1, 6)]
    return pd.concat(parts, ignore_index=True)

def compute_cosine_similarity(query_vec, doc_vecs):
    norms = np.linalg.norm(query_vec) * np.linalg.norm(doc_vecs, axis=1)
    return np.dot(doc_vecs, query_vec) / norms

st.markdown("""
    <style>
    .main-title { font-size: 5em; text-align: center; }
    .main-description { font-size: 2em; text-align: center; }
    .custom-table { width: 100% !important; }
    .custom-table th, .custom-table td { white-space: normal !important; }
    .custom-table a { color: #1f77b4; text-decoration: none; }
    .custom-table a:hover { text-decoration: underline; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>Paper Finder 🔍</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='main-description'>Discover research papers based on your query</h2>", unsafe_allow_html=True)

query_text = st.text_input("Query:")
k = st.number_input("Number of results", min_value=1, max_value=50, value=10)

if st.button("Search"):
    if not query_text.strip():
        st.error("Please enter a valid query.")
    else:
        with st.spinner("Searching for papers..."):
            model, index, embeddings, df = load_model(), load_faiss_index(), load_embeddings(), load_dataframe()
            query_embedding = model.encode([query_text])
            k = min(k, len(df))
            distances, indices = index.search(query_embedding, k)
            valid_indices = [idx for idx in indices[0] if 0 <= idx < len(df)]
            if not valid_indices:
                st.error("No valid results found.")
            else:
                cosine_sims = compute_cosine_similarity(query_embedding.flatten(), embeddings[valid_indices])
                result_df = df.iloc[valid_indices][['full_title', 'abstract', 'doi', 'publication_year']].copy()
                result_df['L2_score'] = [dist for i, dist in zip(indices[0], distances[0]) if i in valid_indices]
                result_df['cosine_sim'] = cosine_sims
                result_df['doi'] = result_df['doi'].apply(lambda x: f'<a href="https://doi.org/{x}" target="_blank">{x}</a>')
                result_df['abstract'] = result_df['abstract'].apply(lambda x: f'<div style="max-height: 170px; overflow-y: auto;">{x}</div>')
                st.markdown(result_df.to_html(escape=False, index=False, classes="custom-table"), unsafe_allow_html=True)
