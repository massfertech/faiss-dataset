import streamlit as st
import numpy as np
import json
from google.cloud import storage
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import re
import nltk
from nltk.corpus import stopwords
from io import BytesIO, StringIO
import asyncio

# Set Streamlit page config
st.set_page_config(layout="wide")

# Load secrets from Streamlit
gcs_credentials = json.loads(st.secrets["google"]["credentials"])
bucket_name = st.secrets["google"]["bucket_name"]

# Ensure stopwords are downloaded once
@st.cache_resource
def download_stopwords():
    nltk.download("stopwords")
    return set(stopwords.words("english"))

stop_words = download_stopwords()

def clean_text(text):
    """
    Clean the text: convert to lowercase, remove special characters,
    and remove English stopwords.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join([word for word in text.split() if word not in stop_words])

# Custom CSS Styling
st.markdown(
    """
    <style>
    .main-title { font-size: 5em; text-align: center; margin-bottom: 0.2em; }
    .main-description { font-size: 2em; text-align: center; margin-bottom: 1em; }
    .custom-table { width: 100% !important; table-layout: auto; }
    .custom-table th, .custom-table td { white-space: normal !important; word-wrap: break-word; }
    .custom-table a { color: #1f77b4; text-decoration: none; }
    .custom-table a:hover { text-decoration: underline; }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title and description
st.markdown("<h1 class='main-title'>Paper Finder 🔍</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='main-description'>Discover the most relevant research papers based on your query</h2>", unsafe_allow_html=True)

st.markdown(
    """
    **How it works?**  
    - Your query is converted into an embedding using a Sentence-BERT model.  
    - The FAISS index retrieves the closest papers using L2 distance.  
    - Cosine similarity is also calculated for an additional similarity metric.
    """
)

# User input
query_text = st.text_input("Query:")
k = st.number_input("Number of results", min_value=1, max_value=50, value=10)

# Cache-heavy resources
@st.cache_resource
def load_model():
    return  SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_resource
def load_faiss_index():
    return faiss.read_index("data/faiss_index (4).bin")

@st.cache_data
def load_embeddings():
    return np.load("data/embeddings (4).npy")

@st.cache_data
def load_dataframe():
    files = [f"data/final_df_21_02_part_{i}.parquet" for i in range(1, 6)]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def load_df_from_gcs(blob_name):
    """Lee un archivo Parquet desde GCS filtrando solo las columnas necesarias."""
    client = storage.Client.from_service_account_info(gcs_credentials)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    content = blob.download_as_bytes()
    required_columns = ['full_title', 'abstract', 'publication_year', 'doi', 'CitedCount']
    return pd.read_parquet(BytesIO(content), columns=required_columns)


# Search functionality
if st.button("Search"):
    if not query_text.strip():
        st.error("Please enter a valid query.")
    else:
        with st.spinner("Searching for papers..."):
            model = load_model()
            index = load_faiss_index()
            embeddings = load_embeddings()
            df = load_dataframe()

            # df = load_df_from_gcs("df0.parquet")
            # st.write("DataFrame Loaded from GCS:", df.head()
            st.write(f"Cant de papers: {len(df)}")

            query_embedding = model.encode([query_text])
            k = min(k, len(df))

            distances, indices = index.search(query_embedding, k)
            query_embedding_flat = query_embedding.flatten()

            valid_indices = [idx for idx in indices[0] if 0 <= idx < len(df)]
            if not valid_indices:
                st.error("No valid results found.")
            else:
                cosine_sims = [
                    np.dot(query_embedding_flat, embeddings[idx])
                    / (np.linalg.norm(query_embedding_flat) * np.linalg.norm(embeddings[idx]))
                    for idx in valid_indices
                ]

                result_df = df.iloc[valid_indices][["full_title", "abstract", "doi", "publication_year"]].copy()
                valid_distances = [dist for i, dist in zip(indices[0], distances[0]) if i in valid_indices]

                result_df["L2_score"] = valid_distances
                result_df["cosine_sim"] = cosine_sims
                result_df = result_df[["full_title", "abstract", "publication_year", "doi", "cosine_sim", "L2_score"]]

                result_df["doi"] = result_df["doi"].apply(lambda x: f'<a href="https://doi.org/{x}" target="_blank">{x}</a>')
                result_df["abstract"] = result_df["abstract"].apply(lambda x: f'<div style="max-height: 170px; overflow-y: auto;">{x}</div>')

                result_df = result_df[:50]

                st.markdown(result_df.to_html(escape=False, index=False, classes="custom-table"), unsafe_allow_html=True)

                st.markdown(
                    """
                    **Explanation of Metrics:**
                    - **Cosine Similarity:** Measures how similar the query and papers are in meaning (closer to 1 is better).
                    - **L2 Score (Euclidean Distance):** Measures the straight-line distance between embeddings (lower is better).
                    - **Which One to Use?** Cosine similarity generally provides better semantic comparisons.
                    """
                )
