import streamlit as st
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import nltk
import re
from nltk.corpus import stopwords

# Descargar stopwords (solo la primera vez)
nltk.download('stopwords')

def clean_text(text):
    """
    Función de limpieza: convierte a minúsculas, elimina caracteres especiales
    y quita stopwords.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Usamos st.cache_resource para objetos pesados y st.cache_data para datos
@st.cache_resource
def load_model():
    # Cargamos el modelo Sentence-BERT
    model = SentenceTransformer('all-mpnet-base-v2')
    return model

@st.cache_resource
def load_faiss_index():
    # Cargamos el índice FAISS preconstruido
    index = faiss.read_index("data/faiss_index.bin")
    return index

@st.cache_data
def load_embeddings():
    # Cargamos los embeddings precomputados
    embeddings = np.load("data/embeddings.npy")
    return embeddings

@st.cache_data
def load_dataframe():
    # Cargamos el DataFrame con la metadata de los papers
    df = pd.read_csv("data/df1_part1.csv")
    return df

# Interfaz de la app
st.title("Buscador de Papers")

# Campo para la consulta
query_text = st.text_input("Ingresa tu consulta:")

# Número de resultados a mostrar
k = st.number_input("Número de resultados", min_value=1, max_value=50, value=10)

if st.button("Buscar"):
    if query_text.strip() == "":
        st.error("Por favor, ingresa una consulta válida.")
    else:
        with st.spinner("Buscando papers..."):
            # Carga de datos y modelos (se hace solo una vez gracias al caching)
            model = load_model()
            index = load_faiss_index()
            embeddings = load_embeddings()
            df = load_dataframe()

            # Convertimos la consulta en un embedding
            query_embedding = model.encode([query_text])
            
            # Buscamos en el índice FAISS los k vecinos más cercanos (usando L2)
            distances, indices = index.search(query_embedding, k)
            query_embedding_flat = query_embedding.flatten()

            # Calculamos la similitud coseno para cada resultado
            cosine_sims = []
            for idx in indices[0]:
                doc_embedding = embeddings[idx]
                dot_product = np.dot(query_embedding_flat, doc_embedding)
                norm_product = np.linalg.norm(query_embedding_flat) * np.linalg.norm(doc_embedding)
                cosine_sims.append(dot_product / norm_product)

            # Creamos el DataFrame de resultados
            result_df = df.iloc[indices[0]][['full_title', 'abstract', 'doi']].copy()
            result_df['L2_score'] = distances[0]
            result_df['cosine_sim'] = cosine_sims
            result_df = result_df[['full_title', 'abstract', 'doi', 'cosine_sim', 'L2_score']]

            st.write(result_df)
