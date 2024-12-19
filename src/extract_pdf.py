from sentence_transformers import SentenceTransformer, util
import chromadb
from chromadb.config import Settings
import fitz
import torch
import ollama
import subprocess

def extract_text(file_path):
    '''Función encargada de extraer solo el texto de un PDF'''
    doc = fitz.open(file_path)
    text = '' 
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text 


def chunk_text(text, chunk_size=30):
    '''División en chunks del texto'''
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


def connect_to_chroma():
    client = chromadb.Client() 
    return client

def create_or_get_collection(client, collection_name="pdf_chunks"):
    """Crear o acceder a una colección de Chroma"""
    if collection_name not in client.list_collections():
        collection = client.create_collection(collection_name, metadata={"hnsw:space":"cosine"})
        print(f"Colección '{collection_name}' creada.")
    else:
        collection = client.get_collection(collection_name)
        print(f"Colección '{collection_name}' encontrada.")
    return collection


def generate_embeddings(chunks, model):
    """Generar embeddings para los chunks de texto"""
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings


def store_embeddings_in_chroma(collection, embeddings, chunks):
    """Almacenar los embeddings en Chroma"""
    points = [
        {"id": str(i), "embedding": embedding, "metadata": {"chunk": chunk}}
        for i, (embedding, chunk) in enumerate(zip(embeddings, chunks))
    ]
    collection.upsert(
        documents=[point["metadata"]["chunk"] for point in points],
        metadatas=[point["metadata"] for point in points],
        embeddings=[point["embedding"] for point in points],
        ids=[point["id"] for point in points]
    )
    print(f"{len(points)} puntos insertados en Chroma.")


def search_in_chroma(collection, query_embedding, top_k=1):
    """Buscar en Chroma los contextos más relevantes"""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results


def generate_natural_language_response(results, question, model_name='llama3.2'):
    """Generar una respuesta en lenguaje natural usando Ollama"""
    #client = ollama.Client()
    metadatas = results['metadatas'][0] 
    texts = [metadata['chunk'] for metadata in metadatas]
    prompt = f"Basado en el siguiente texto:\n\n{' '.join(texts)}\n\nResponde a la siguiente pregunta: {question}"
    
    result = subprocess.run(['ollama','run','llama3.2', prompt], capture_output=True)
    #response = client.chat(model_name, messages=[{'role': 'user', 'content': prompt}])
    
    #print("Respuesta de Ollama:", response)
    print("Respuesta de Ollama:", result.stdout.strip())
    return result.stdout.strip()#response


def main(file_path, collection_name="pdf_chunks", top_k=3):
    # Paso 1: Cargar texto y dividir en chunks (ajustar según tu necesidad)
    pdf_text = extract_text(file_path) 
    pdf_chunks = chunk_text(pdf_text) 

    # Paso 2: Generar embeddings
    model_name='all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    embeddings = generate_embeddings(pdf_chunks, model)

    # Paso 3: Conectar a Chroma y crear la colección si no existe
    client = connect_to_chroma()
    collection = create_or_get_collection(client, collection_name)

    # Paso 4: Almacenar embeddings en Chroma
    store_embeddings_in_chroma(collection, embeddings, pdf_chunks)

    # Paso 5: Definir la consulta del usuario
    question = "disposiciones del ejecutivo con rango de ley"
    question_embedding = generate_embeddings([question], model)[0] 

    # Paso 6: Buscar contextos relevantes
    results = search_in_chroma(collection, question_embedding, top_k)
    print(len(results), type(results))

    # Paso 7: Llamar a Ollama
    response = generate_natural_language_response(results, question, model_name='llama3.2')
    print("Respuesta generada:", response)

# Llamada a la función principal con el archivo PDF
if __name__ == "__main__":
    file_path = r'C:\Users\ldebe\Downloads\Tema_Derecho.pdf'
    main(file_path)