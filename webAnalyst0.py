import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

# Función para calcular el TF-IDF de un conjunto de párrafos
def calcular_tfidf(parrafos):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(parrafos)
    tfidf_scores = vectorizer.idf_
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_scores, feature_names

# Función para filtrar palabras clave comunes
def filtrar_palabras_clave(palabras_clave):
    # Lista de palabras comunes que deseamos excluir (artículos, conjunciones, preposiciones, etc.)
    palabras_excluidas = set(stopwords.words('english'))
    palabras_clave_filtradas = [palabra for palabra in palabras_clave if palabra.lower() not in palabras_excluidas and not palabra.isdigit()]
    return palabras_clave_filtradas

# Función para obtener la descripción de la importancia de la palabra clave
def obtener_descripcion(palabra, contenido):
    # Aquí puedes utilizar técnicas de NLP para analizar el contenido y generar descripciones basadas en contexto
    # Por ejemplo, puedes buscar el uso de la palabra en el contenido y extraer frases relevantes

    descripcion = ""
    palabra = palabra.lower()  # Convertir la palabra a minúsculas para una búsqueda sin distinción entre mayúsculas y minúsculas

    # Analizar el contenido en busca de frases que contengan la palabra
    for parrafo in contenido:
        if palabra in parrafo.lower():
            # Si se encuentra la palabra en el párrafo, puedes extraer una frase relevante
            # Aquí, se toma una oración que contiene la palabra clave
            oraciones = nltk.sent_tokenize(parrafo)
            for oracion in oraciones:
                if palabra in oracion.lower():
                    descripcion = oracion
                    break  # Detener la búsqueda después de encontrar la primera coincidencia

    return descripcion

# Función para analizar un sitio web
def analizar_sitio_web(url, tipo_contenido, variables_interes):
    try:
        # Realizar una solicitud HTTP para obtener el contenido de la página
        response = requests.get(url)
        response.raise_for_status()  # Verificar si la solicitud fue exitosa

        # Utilizar BeautifulSoup para analizar el contenido de la página
        soup = BeautifulSoup(response.text, 'html.parser')

        # Obtener el título de la página
        title = soup.title.string if soup.title else "No se encontró título"

        # Obtener las metaetiquetas (meta tags)
        meta_tags = soup.find_all('meta')

        # Buscar contenido de calidad y relevante
        contenido = soup.find_all('p')  # Buscar párrafos como ejemplo de contenido

        # Calcular el TF-IDF para los párrafos y obtener las palabras clave
        parrafos = [p.get_text() for p in contenido]
        tfidf_scores, feature_names = calcular_tfidf(parrafos)

        # Seleccionar los párrafos más relevantes (con puntuación alta)
        parrafos_relevantes = [parrafos[i] for i, score in enumerate(tfidf_scores) if i < len(parrafos) and score > 0.2]

        # Filtrar palabras clave comunes
        palabras_clave_filtradas = filtrar_palabras_clave(feature_names)

        # Crear un DataFrame de pandas para mostrar las palabras clave y sus puntuaciones
        df = pd.DataFrame({'Palabra Clave': palabras_clave_filtradas[:20], 'Puntuación TF-IDF': tfidf_scores[:20]})

        # Añadir descripciones a las palabras clave
        df['Descripción'] = df.apply(lambda row: obtener_descripcion(row['Palabra Clave'], parrafos), axis=1)

        # Ordenar el DataFrame por puntuación TF-IDF de mayor a menor
        df = df.sort_values(by='Puntuación TF-IDF', ascending=False)

        # Imprimir los resultados
        print("Análisis del sitio web:", url)
        print("Tipo de Contenido de Interés:", tipo_contenido)
        print("Variables de Interés:", variables_interes)
        print("Estado de la página:", response.status_code)
        print("Título de la página:", title)

        print("\nMetaetiquetas:")
        for tag in meta_tags:
            print(f"{tag.get('name', '')}: {tag.get('content', '')}")

        print("\nContenido de calidad y relevante:")
        for parrafo in parrafos_relevantes:
            print(parrafo)

        # Imprimir las palabras clave y sus puntuaciones en forma de tabla
        print("\nPalabras clave detectadas (TF-IDF):\n")
        print(df.to_string(index=False))

    except requests.exceptions.RequestException as e:
        print("Error al realizar la solicitud HTTP:", e)

if __name__ == "__main__":
    url = input("Por favor, ingresa la URL del sitio web que deseas analizar: ")
    tipo_contenido = input("Tipo de Contenido (comercial, político, financiero, social, ambiental, histórico, tecnología, legal, científico): ")
    variables_interes = input("Variables de Interés (separadas por comas): ").split(", ")
    analizar_sitio_web(url, tipo_contenido, variables_interes)
