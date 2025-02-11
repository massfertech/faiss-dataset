import streamlit as st
import pandas as pd
import numpy as np
import faiss
import nltk
import re
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords

# Configuración de la página
st.set_page_config(page_title="Buscador Científico", page_icon="🔬")
st.title("Búsqueda Semántica en Artículos Científicos")
st.write("Escribe tu pregunta o tema de interés para encontrar los papers más relevantes:")

# Cargar datos preprocesados
@st.cache_data
def cargar_metadata():
    return pd.read_csv("data/df1_part1.csv")

@st.cache_data
def cargar_embeddings():
    return np.load("data/embeddings.npy")

@st.cache_resource
def cargar_indice_faiss():
    return faiss.read_index("data/faiss_index.bin")

@st.cache_resource
def cargar_modelo():
    return SentenceTransformer('all-mpnet-base-v2')

# Cargar todos los componentes
with st.spinner('Cargando sistema de búsqueda...'):
    df = cargar_metadata()
    embeddings = cargar_embeddings()
    indice_faiss = cargar_indice_faiss()
    modelo = cargar_modelo()

# Widget de búsqueda
consulta = st.text_input("**Escribe tu consulta científica:**", 
                       placeholder="Ej: Efectos del COVID-19 en el sistema cardiovascular...")

if consulta:
    with st.spinner(f'Buscando en {len(df)} artículos...'):
        # Generar embedding para la consulta
        query_embedding = modelo.encode([consulta])
        
        # Búsqueda FAISS
        k = 10
        distancias, indices = indice_faiss.search(query_embedding, k)
        
        # Calcular similitud coseno
        query_flat = query_embedding.flatten()
        similitudes = []
        for idx in indices[0]:
            doc_embedding = embeddings[idx]
            dot_product = np.dot(query_flat, doc_embedding)
            norm_product = np.linalg.norm(query_flat) * np.linalg.norm(doc_embedding)
            similitudes.append(dot_product / norm_product)
        
        # Crear DataFrame de resultados
        resultados = df.iloc[indices[0]].copy()
        resultados['L2_score'] = distancias[0]
        resultados['cosine_sim'] = similitudes
        resultados = resultados[['full_title', 'abstract', 'doi', 'cosine_sim', 'L2_score']]

    # Mostrar resultados
    st.subheader(f"📚 Top {k} resultados para: '{consulta}'")
    
    for idx, fila in resultados.iterrows():
        with st.expander(f"{fila['full_title']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Similitud Coseno", f"{fila['cosine_sim']:.1%}")
            with col2:
                st.metric("Distancia L2", f"{fila['L2_score']:.2f}")
            
            st.write("**DOI:**", fila['doi'])
            st.write("**Abstract:**", fila['abstract'])
            
    st.success(f"Búsqueda completada! Resultados ordenados por relevancia.")

else:
    st.info("💡 Ejemplo de consulta: 'Avances recientes en terapia génica para diabetes tipo 2'")

# Sidebar con información
with st.sidebar:
    st.header("⚙️ Información del Sistema")
    st.write(f"**Base de datos:** {len(df)} artículos científicos")
    st.write("**Modelo de embeddings:** all-mpnet-base-v2")
    st.write("**Métricas de búsqueda:**")
    st.write("- Similitud coseno (0-1, mayor es mejor)")
    st.write("- Distancia L2 (menor es mejor)")
    
    if st.checkbox("Mostrar metadatos técnicos"):
        st.write("**Dimensiones embeddings:**", embeddings.shape[1])
        st.write("**Tipo de índice FAISS:**", indice_faiss.__class__.__name__)
