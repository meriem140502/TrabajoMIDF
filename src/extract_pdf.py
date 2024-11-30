import fitz
import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

def extract_text_and_images(file_path):
    ''' Funcion encargada de extraer la información del pdf'''
    doc = fitz.open(file_path)
    text = ''
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
        images.extend(page.get_images(full=True))
    return text, images

def chunk_text(text, chunk_size=75):
    ''' División en chunks del texto'''
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def generate_embeddings(chunks, model_name='all-MiniLM-L6-v2'):
    ''' Genera embeddings para los chunks de texto '''
    model = SentenceTransformer(model_name)  # Carga el modelo preentrenado
    embeddings = model.encode(chunks, show_progress_bar=True)  # Genera embeddings para cada chunk
    
    # Imprimir los embeddings
    for i, embedding in enumerate(embeddings):
        print(f"Embedding {i}: {embedding}")
    
    return embeddings

def store_embeddings_in_qdrant(chunks, embeddings, qdrant_client, collection_name="pdf_chunks"):
    ''' Almacena los embeddings en Qdrant '''
    if not qdrant_client.collection_exists(collection_name):
        print(f"Collection {collection_name} not found. Creating collection.")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={"size": len(embeddings[0]), "distance": "Cosine"}  # Dimensión del vector y distancia
        )
    else:
        print(f"Collection {collection_name} exists.")
    
    # Agregar los embeddings como puntos en la colección
    points = [
        PointStruct(
            id=i,  # ID único para cada chunk
            vector=embedding.tolist(),  # El embedding generado
            payload={"chunk": chunk}  # Metadatos asociados (el texto original)
        )
        for i, (embedding, chunk) in enumerate(zip(embeddings, chunks))
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)
    
if __name__ == '__main__':
    # ruta del archivo PDF
    file_path = r'C:\Users\ldebe\Downloads\CV.pdf'
    
    # Cargar y trocear el PDF
    pdf_text, pdf_images = extract_text_and_images(file_path)
    pdf_chunks = chunk_text(pdf_text)

    # Paso 2: Generar embeddings para los chunks
    embeddings = generate_embeddings(pdf_chunks)

    # Paso 3: Conectar a Qdrant
    qdrant_client = QdrantClient(":memory:")  # Usa Qdrant en memoria (para pruebas locales)
    # Usa `QdrantClient(host="localhost", port=6333)` si tienes un servidor Qdrant corriendo

    # Paso 4: Almacenar embeddings en Qdrant
    store_embeddings_in_qdrant(pdf_chunks, embeddings, qdrant_client)

    print(f"Embeddings generados y almacenados en Qdrant. Total chunks: {len(pdf_chunks)}")