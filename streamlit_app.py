import streamlit as st
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import nltk
import re
from nltk.corpus import stopwords

# Descargamos stopwords (esto se hace solo la primera vez)
nltk.download('stopwords')

# Si bien en este ejemplo la función de limpieza no se usa en la consulta,
# la incluí en caso de que quieras aplicarla en el futuro a la query o a nuevos textos.
def clean_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r'[^\w\s]', '', text)  # Eliminar caracteres especiales
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Usamos los decoradores de cache para que estos objetos pesados se carguen solo una vez
# Nota: En versiones recientes de Streamlit, se recomienda st.cache_resource para objetos como modelos
@st.cache_resource
def load_model():
    # Cargamos el modelo de Sentence-BERT (puedes elegir el que mejor se ajuste a tus necesidades)
    model = SentenceTransformer('all-mpnet-base-v2')
    return model

@st.cache_resource
def load_faiss_index():
    # Cargamos el índice FAISS previamente guardado
    index = faiss.read_index("/data/fais_index.bin")
    return index

@st.cache_data
def load_embeddings():
    # Cargamos los embeddings preprocesados
    embeddings = np.load("data/embeddings.npy")
    return embeddings

@st.cache_data
def load_dataframe():
    # Cargamos el DataFrame con la información de los papers
    df = pd.read_csv("/data/df1_part1.csv")
    return df

# Interfaz de la app
st.title("Buscador de Papers")

query_text = st.text_input("Ingresa tu consulta:")
k = st.number_input("Número de resultados", min_value=1, max_value=50, value=10)

if st.button("Buscar"):
    if query_text.strip() == "":
        st.error("Por favor, ingresa una consulta válida.")
    else:
        with st.spinner("Buscando papers..."):
            # Cargamos los objetos (esto se hace solo la primera vez gracias al caching)
            model = load_model()
            index = load_faiss_index()
            embeddings = load_embeddings()
            df = load_dataframe()

            # Obtenemos el embedding de la consulta
            query_embedding = model.encode([query_text])
            
            # Buscamos en el índice FAISS los k vecinos más cercanos (usando L2)
            distances, indices = index.search(query_embedding, k)
            query_embedding_flat = query_embedding.flatten()

            # Calculamos la similitud coseno manualmente para cada resultado
            cosine_sims = []
            for idx in indices[0]:
                doc_embedding = embeddings[idx]
                dot_product = np.dot(query_embedding_flat, doc_embedding)
                norm_product = np.linalg.norm(query_embedding_flat) * np.linalg.norm(doc_embedding)
                cosine_sims.append(dot_product / norm_product)

            # Creamos el DataFrame de resultados con las columnas deseadas
            # Nos aseguramos de que el DataFrame original tenga las columnas 'full_title', 'abstract' y 'doi'
            result_df = df.iloc[indices[0]][['full_title', 'abstract', 'doi']].copy()
            result_df['L2_score'] = distances[0]
            result_df['cosine_sim'] = cosine_sims

            # Reordenamos las columnas
            result_df = result_df[['full_title', 'abstract', 'doi', 'cosine_sim', 'L2_score']]

            st.write(result_df)
