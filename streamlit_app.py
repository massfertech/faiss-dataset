import streamlit as st
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import nltk
import re
from nltk.corpus import stopwords

# Download stopwords (only once)
nltk.download('stopwords')

def clean_text(text):
    """
    Clean the text: convert to lowercase, remove special characters,
    and remove English stopwords.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Custom CSS for styling
st.markdown("""
    <style>
    /* Title and description styling */
    .main-title {
        font-size: 5em;
        text-align: center;
        margin-bottom: 0.2em;
    }
    .main-description {
        font-size: 2em;
        text-align: center;
        margin-bottom: 1em;
    }
    /* Table styling */
    div[data-baseweb="table"] {
        width: 100% !important;
    }
    div[data-baseweb="table"] table {
        width: 100% !important;
        table-layout: auto;
    }
    /* Force full wrapping and avoid truncation */
    div[data-baseweb="table"] table thead tr th, 
    div[data-baseweb="table"] table tbody tr td {
        white-space: normal !important;
        word-wrap: break-word;
    }
    /* Increase minimum width for the first column (full_title) */
    div[data-baseweb="table"] table tbody tr td:nth-child(1),
    div[data-baseweb="table"] table thead tr th:nth-child(1) {
        min-width: 300px;
    }
    </style>
""", unsafe_allow_html=True)

# Big title and description
st.markdown("<h1 class='main-title'>Paper Finder 🔍</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='main-description'>Discover the most relevant research papers based on your query</h2>", unsafe_allow_html=True)

# "How it works" description (unchanged)
st.markdown("""
**How it works?**  
- Your query is converted into an embedding using a Sentence-BERT model.  
- The FAISS index is used to retrieve the closest papers based on the L2 distance.  
- Additionally, cosine similarity is calculated to provide another similarity metric.
""")

# Text above the search bar
st.markdown("**Enter your search below and enjoy exploring the research papers:**")

# Use st.cache_resource for heavy objects and st.cache_data for data
@st.cache_resource
def load_model():
    # Load the Sentence-BERT model
    model = SentenceTransformer('all-mpnet-base-v2')
    return model

@st.cache_resource
def load_faiss_index():
    # Load the pre-built FAISS index
    index = faiss.read_index("data/faiss_index.bin")
    return index

@st.cache_data
def load_embeddings():
    # Load precomputed embeddings
    embeddings = np.load("data/embeddings.npy")
    return embeddings

@st.cache_data
def load_dataframe():
    # Load the metadata for the papers by combining four CSV files.
    # This allows you to bypass GitHub's file size limitations by splitting your dataset.
    df1 = pd.read_csv("data/final_df_part_1.csv", encoding="utf-8")
    df2 = pd.read_csv("data/final_df_part_2.csv")
    df3 = pd.read_csv("data/final_df_part_3.csv")
    df4 = pd.read_csv("data/final_df_part_4.csv")
    df = pd.concat([df1, df2, df3, df4], ignore_index=True)
    return df

# User input for query and number of results to display
query_text = st.text_input("Query:")
k = st.number_input("Number of results", min_value=1, max_value=50, value=10)

if st.button("Search"):
    if query_text.strip() == "":
        st.error("Please enter a valid query.")
    else:
        with st.spinner("Searching for papers..."):
            # Load models and data (cached, so loaded only once)
            model = load_model()
            index = load_faiss_index()
            embeddings = load_embeddings()
            df = load_dataframe()
            
            # Convert the query into an embedding
            query_embedding = model.encode([query_text])
            
            # Ensure that k does not exceed the number of papers in the dataset
            k = min(k, len(df))
            
            # Search the FAISS index for the k nearest neighbors using L2 distance
            distances, indices = index.search(query_embedding, k)
            query_embedding_flat = query_embedding.flatten()
            
            # Filter valid indices (avoid negative or out-of-bound indices)
            valid_indices = [idx for idx in indices[0] if idx >= 0 and idx < len(df)]
            if not valid_indices:
                st.error("No valid results found.")
            else:
                # Calculate cosine similarity for each valid result
                cosine_sims = []
                for idx in valid_indices:
                    doc_embedding = embeddings[idx]
                    dot_product = np.dot(query_embedding_flat, doc_embedding)
                    norm_product = np.linalg.norm(query_embedding_flat) * np.linalg.norm(doc_embedding)
                    cosine_sims.append(dot_product / norm_product)
            
                # Extract the corresponding rows from the DataFrame
                result_df = df.iloc[valid_indices][['full_title', 'abstract', 'doi']].copy()
                
                # Filter the distances corresponding to the valid indices
                valid_distances = [dist for i, dist in zip(indices[0], distances[0]) if i in valid_indices]
                
                # Add similarity metrics to the DataFrame
                result_df['L2_score'] = valid_distances
                result_df['cosine_sim'] = cosine_sims
                result_df = result_df[['full_title', 'abstract', 'doi', 'cosine_sim', 'L2_score']]
                
                # Optional: Calculate dynamic table height if desired
                table_height = (len(result_df) + 1) * 35 + 3
                
                # Display the DataFrame using the full container width and dynamic height
                st.dataframe(result_df, use_container_width=True, height=table_height)
