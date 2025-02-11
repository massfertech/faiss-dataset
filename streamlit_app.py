import streamlit as st
import pandas as pd
import numpy as np
import faiss
import nltk
import re
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords

# Configuración inicial de la página
st.set_page_config(page_title="Buscador de Papers Científicos", page_icon="📚")
st.title("Búsqueda Semántica en Artículos Científicos")
st.write("""
Busca artículos relevantes usando lenguaje natural. 
El sistema entenderá el contexto de tu búsqueda y encontrará los papers más relevantes.
""")

# Descargar stopwords de NLTK
@st.cache_data
def descargar_stopwords():
    nltk.download('stopwords')
descargar_stopwords()
stop_words = set(stopwords.words('english'))

# Función para limpiar texto
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = ' '.join([palabra for palabra in texto.split() if palabra not in stop_words])
    return texto

# Cargar modelo de Sentence Transformers
@st.cache_resource
def cargar_modelo():
    return SentenceTransformer('all-mpnet-base-v2')

modelo = cargar_modelo()

# Cargar y preprocesar datos
@st.cache_data
def cargar_datos():
    datos = pd.read_csv("data/articulos_cientificos.csv")
    datos['texto_completo'] = datos['abstract'] + " " + datos['full_text']
    datos['texto_limpio'] = datos['texto_completo'].apply(limpiar_texto)
    return datos

datos = cargar_datos()

# Generar embeddings y crear índice FAISS
@st.cache_resource
def preparar_sistema_busqueda(datos):
    textos = datos['texto_limpio'].tolist()
    embeddings = modelo.encode(textos)
    dimension = embeddings.shape[1]
    indice = faiss.IndexFlatL2(dimension)
    indice.add(embeddings)
    return embeddings, indice

embeddings, indice_faiss = preparar_sistema_busqueda(datos)

# Widget de búsqueda
consulta = st.text_input("Escribe tu consulta científica:", 
                       placeholder="Ej: Avances recientes en terapia génica para cáncer...")

if consulta:
    with st.spinner('Buscando en más de 10,000 artículos...'):
        # Generar embedding para la consulta
        embedding_consulta = modelo.encode([consulta])
        
        # Búsqueda en el índice FAISS
        k = 10
        distancias, indices = indice_faiss.search(embedding_consulta, k)
        
        # Calcular similitud coseno
        embedding_plano = embedding_consulta.flatten()
        similitudes = []
        for idx in indices[0]:
            embedding_articulo = embeddings[idx]
            producto_punto = np.dot(embedding_plano, embedding_articulo)
            norma = np.linalg.norm(embedding_plano) * np.linalg.norm(embedding_articulo)
            similitudes.append(producto_punto / norma)
        
        # Crear DataFrame con resultados
        resultados = datos.iloc[indices[0]][['titulo', 'abstract', 'doi', 'año_publicacion']]
        resultados['Similitud'] = similitudes
        resultados['Distancia'] = distancias[0]
        
        # Formatear resultados
        resultados = resultados.sort_values('Similitud', ascending=False)
        resultados['Similitud'] = resultados['Similitud'].apply(lambda x: f"{x:.1%}")
        resultados['Distancia'] = resultados['Distancia'].apply(lambda x: f"{x:.2f}")

    # Mostrar resultados
    st.subheader(f"Top {k} resultados para: '{consulta}'")
    
    for _, fila in resultados.iterrows():
        with st.expander(f"📄 {fila['titulo']} ({fila['año_publicacion']})"):
            st.markdown(f"**DOI:** `{fila['doi']}`")
            st.markdown(f"**Similitud:** {fila['Similitud']} | **Distancia:** {fila['Distancia']}")
            st.markdown("**Resumen:**")
            st.write(fila['abstract'])
            
    st.success(f"¡Búsqueda completada! Encontrados {len(resultados)} resultados relevantes.")
else:
    st.info("💡 Escribe una consulta en el campo superior para comenzar tu búsqueda.")

# Sección adicional con información del dataset
with st.sidebar:
    st.header("⚙️ Acerca del sistema")
    st.markdown("""
    **Tecnologías utilizadas:**
    - Modelo de embeddings: `all-mpnet-base-v2`
    - Búsqueda semántica: FAISS
    - Procesamiento de texto: NLTK y RegEx
    
    **Características del dataset:**
    - Artículos científicos de PubMed/PMC
    - Período: 2010-2023
    - Campos incluidos: título, resumen, texto completo, DOI
    """)
    
    if st.checkbox("Mostrar metadatos del dataset"):
        st.write(f"📊 Total de artículos: {len(datos):,}")
        st.write(f"📅 Rango temporal: {datos['año_publicacion'].min()}-{datos['año_publicacion'].max()}")
        st.write("🔠 Campos disponibles:", ", ".join(datos.columns))
