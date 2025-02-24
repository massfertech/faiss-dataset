import streamlit as st
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import nltk
import re
from nltk.corpus import stopwords
import rarfile
import io
import patoolib
import tempfile
import os

# Set page layout to wide
st.set_page_config(layout="wide")

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
    /* Table styling: force full width and word-wrap */
    .custom-table {
        width: 100% !important;
        table-layout: auto;
    }
    .custom-table th, .custom-table td {
        white-space: normal !important;
        word-wrap: break-word;
    }
    /* Optional: adjust link color if needed */
    .custom-table a {
        color: #1f77b4;
        text-decoration: none;
    }
    .custom-table a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# Big title and description
st.markdown("<h1 class='main-title'>Paper Finder 🔍</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='main-description'>Discover the most relevant research papers based on your query</h2>", unsafe_allow_html=True)

# "How it works" description
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
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

@st.cache_resource
def load_faiss_index():
    # Load the pre-built FAISS index
    index = faiss.read_index("data/faiss_index (3).bin")
    return index

@st.cache_data
def load_embeddings():
    # Load precomputed embeddings
    embeddings = np.load("data/embeddings (3).npy")
    return embeddings

@st.cache_data
def load_dataframe():
    # Lista de archivos RAR con los CSV
    rar_files = [
        "data/df_part_1.rar",
        "data/df_part_2.rar",
        "data/df_part_3.rar",
        "data/df_part_4.rar"
    ]
    
    df_list = []
    # Ruta al ejecutable de unrar (asegúrate de que tenga permisos de ejecución)
    unrar_path = os.path.join("bin", "7za")
    
    for rf in rar_files:
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Extrae el contenido del archivo RAR en el directorio temporal, usando el ejecutable especificado
            patoolib.extract_archive(rf, outdir=tmpdirname, program=unrar_path)
            # Se asume que hay un único archivo CSV en cada RAR.
            for file in os.listdir(tmpdirname):
                if file.endswith('.csv'):
                    csv_path = os.path.join(tmpdirname, file)
                    df = pd.read_csv(csv_path)
                    df_list.append(df)
                    # Opcionalmente, elimina el archivo extraído
                    os.remove(csv_path)
    # Concatenar todos los DataFrames
    df_full = pd.concat(df_list, ignore_index=True)
    print("Total papers:", len(df_full))
    return df_full

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
            print(len(df))
            
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
                result_df = df.iloc[valid_indices][['full_title', 'abstract', 'doi', 'publication_year']].copy()
                
                # Filter the distances corresponding to the valid indices
                valid_distances = [dist for i, dist in zip(indices[0], distances[0]) if i in valid_indices]
                
                # Add similarity metrics to the DataFrame
                result_df['L2_score'] = valid_distances
                result_df['cosine_sim'] = cosine_sims
                result_df = result_df[['full_title', 'abstract', 'publication_year', 'doi', 'cosine_sim', 'L2_score']]
                
                # Convert the 'doi' column into clickable links
                result_df['doi'] = result_df['doi'].apply(
                    lambda x: f'<a href="https://doi.org/{x}" target="_blank">{x}</a>'
                )
                
                # Limit the text in the 'abstract' column with a scrollable div.
                # Ajusta "150px" según la cantidad de texto que quieras mostrar por defecto.
                result_df['abstract'] = result_df['abstract'].apply(
                    lambda x: f'<div style="max-height: 170px; overflow-y: auto;">{x}</div>'
                )
                
                # Convert the DataFrame to HTML (with escape=False to allow HTML in the DOI and abstract columns)
                html_table = result_df.to_html(escape=False, index=False, classes="custom-table")
                
                # Instead of wrapping the table in a fixed height container,
                # we display it directly so that it expands vertically to show all rows.
                st.markdown(html_table, unsafe_allow_html=True)
                
                st.markdown("""
                **Explanation of Metrics:**

                - **Cosine Similarity:**  
                  This measures the cosine of the angle between the vector representations of your query and each paper.  
                  A value close to 1 indicates that the texts are very similar, while a value close to 0 indicates less similarity.

                - **L2 Score (Euclidean Distance):**  
                  This calculates the straight-line distance between the vector representations.  
                  Lower values mean the texts are more similar, while higher values suggest they are more different.

                - **Which One to Use?**
                  For evaluating how similar a text is to a query, cosine similarity is generally more relevant because it focuses on the semantic relationship—how the words and their meanings align—rather than the absolute size of the vector. 
                  Although both metrics can provide useful, complementary insights, cosine similarity tends to capture text similarity more effectively in most NLP applications.
                """)
