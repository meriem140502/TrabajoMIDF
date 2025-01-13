from sentence_transformers import SentenceTransformer, util
import chromadb
from chromadb.config import Settings
import fitz
import requests
import json


def extract_text(file_path):
    '''Función encargada de extraer solo el texto de un PDF'''
    doc = fitz.open(file_path)
    text = ''  # Variable para almacenar el texto
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()  # Extraer el texto de cada página
    return text  # Solo devolver el texto



def chunk_text(text, chunk_size=20):
    '''División en chunks del texto'''
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


def generate_embeddings(chunks, model_name='all-MiniLM-L6-v2'):
    '''Genera embeddings para los chunks de texto'''
    model = SentenceTransformer(model_name)  # Carga el modelo preentrenado
    embeddings = model.encode(chunks, show_progress_bar=True)  # Genera embeddings para cada chunk
    
    # Imprimir los embeddings
    for i, embedding in enumerate(embeddings):
        print(f"Embedding {i}: {embedding}")
    
    return embeddings


def connect_to_chroma():
    client = chromadb.Client()  # Usar el constructor sin necesidad de configuraciones antiguas
    return client


def create_or_get_collection(client, collection_name="pdf_chunks"):
    """Crear o acceder a una colección de Chroma"""
    if collection_name not in client.list_collections():
        collection = client.create_collection(collection_name)
        print(f"Colección '{collection_name}' creada.")
    else:
        collection = client.get_collection(collection_name)
        print(f"Colección '{collection_name}' encontrada.")
    return collection



def store_embeddings_in_chroma(collection, embeddings, chunks):
    """Almacenar los embeddings en Chroma"""
    points = [
        {"id": str(i), "embedding": embedding, "metadata": {"chunk": chunk}}
        for i, (embedding, chunk) in enumerate(zip(embeddings, chunks))
    ]
    collection.add(
        documents=[point["metadata"]["chunk"] for point in points],
        metadatas=[point["metadata"] for point in points],
        embeddings=[point["embedding"] for point in points],
        ids=[point["id"] for point in points]
    )
    print(f"{len(points)} puntos insertados en Chroma.")

def search_in_chroma(collection, query_embedding, top_k=3):
    """Buscar en Chroma los contextos más relevantes"""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results

def query_ollama(prompt, model="llama-3"):
    """Enviar una consulta al modelo Ollama y obtener la respuesta"""
    url = "http://localhost:11434/api/chat"  # Cambiar si tienes otro puerto configurado
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response.raise_for_status()  # Verificar errores
    return response.json()["response"]




def main(file_path, collection_name="pdf_chunks", top_k=3):
    # Paso 1: Cargar texto y dividir en chunks
    pdf_text = extract_text(file_path)
    pdf_chunks = chunk_text(pdf_text)

    # Paso 2: Generar embeddings
    embeddings = generate_embeddings(pdf_chunks)

    # Paso 3: Conectar a Chroma y crear la colección si no existe
    client = connect_to_chroma()
    collection = create_or_get_collection(client, collection_name)

    # Paso 4: Almacenar embeddings en Chroma
    store_embeddings_in_chroma(collection, embeddings, pdf_chunks)

    # Paso 5: Definir la consulta del usuario
    question = "¿Qué habilidades tengo?"
    question_embedding = generate_embeddings([question])[0]

    # Paso 6: Buscar contextos relevantes
    results = search_in_chroma(collection, question_embedding, top_k)

    # Concatenar los contextos más relevantes
    relevant_contexts = "\n".join(results['documents'])
    prompt = f"Contexto:\n{relevant_contexts}\n\nPregunta: {question}\n\nRespuesta:"
    
    # Paso 7: Consultar a Ollama
    response = query_ollama(prompt)
    print(f"Respuesta del modelo: {response}")

# Llamar a la función principal con Ollama
if __name__ == "__main__":
    file_path = r'C:\Users\Usuario\Desktop\MIDF\TICs\TrabajoMIDF\src\CV_Meryem.pdf'
    main(file_path)
