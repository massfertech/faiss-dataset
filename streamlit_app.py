import altair as alt
import pandas as pd
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Configurar la página
st.set_page_config(page_title="Pubmed Search Engine", page_icon="🔍")
st.title(" Pubmed Semantic Search")
st.write(
    """
    This app helps you find relevant scientific papers from Pubmed/PMC using AI-powered semantic search.
    Enter your query in natural language and find the most relevant papers!
    """
)

# Cargar modelo una vez y mantenerlo en cache
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Cargar datos y índices FAISS
@st.cache_resource
def load_index():
    index = faiss.read_index("data/faiss_index.bin")
    final_df = pd.read_csv("data/df1_part1.csv")
    embeddings = np.load("data/embeddings.npy")
    return index, final_df, embeddings

model = load_model()
index, final_df, embeddings = load_index()

# Barra de búsqueda
query_text = st.text_input(
    "Enter your scientific query:",
    placeholder="e.g., Can Breastfeeding Prevent Obesity?",
    help="Type your question or keywords and press Enter"
)

if query_text:
    # Generar embedding de la consulta
    query_embedding = model.encode([query_text])
    
    # Búsqueda FAISS
    k = 10  # Número de resultados
    distances, indices = index.search(query_embedding, k)
    
    # Calcular similitud coseno
    query_embedding_flat = query_embedding.flatten()
    cosine_sims = []
    for idx in indices[0]:
        doc_embedding = embeddings[idx]
        dot_product = np.dot(query_embedding_flat, doc_embedding)
        norm_product = np.linalg.norm(query_embedding_flat) * np.linalg.norm(doc_embedding)
        cosine_sims.append(dot_product / norm_product)
    
    # Crear DataFrame de resultados
    result_df = final_df.iloc[indices[0]][['full_title', 'doi', 'abstract','full_text']].copy()
    result_df['Relevance Score'] = cosine_sims
    result_df['L2 Distance'] = distances[0]
    
    # Formatear resultados
    result_df = result_df[['full_title', 'abstract', 'doi', 'Relevance Score', 'L2 Distance']]
    result_df['Relevance Score'] = result_df['Relevance Score'].apply(lambda x: f"{x:.1%}")
    result_df['L2 Distance'] = result_df['L2 Distance'].apply(lambda x: f"{x:.2f}")

    # Mostrar resultados
    st.subheader(f"Top {k} most relevant papers:")
    for idx, row in result_df.iterrows():
        with st.expander(f"{row['full_title']} (Relevance: {row['Relevance Score']})"):
            st.markdown(f"**DOI:** {row['doi']}")
            st.markdown(f"**Abstract:** {row['abstract']}")
            st.markdown(f"**Full text available:** {'' if pd.isna(row['full_text']) else 'Yes'}")
