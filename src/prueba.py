from sentence_transformers import SentenceTransformer, util
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import fitz
import torch
import chromadb


def extract_text_and_images(file_path):
    '''Función encargada de extraer la información del PDF'''
    doc = fitz.open(file_path)
    text = ''
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
        images.extend(page.get_images(full=True))
    return text, images


def chunk_text(text, chunk_size=75):
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







if __name__ == '__main__':
    # Ruta del archivo PDF
    #file_path = r'C:\Users\Usuario\Desktop\MIDF\TICs\TrabajoMIDF\src\CV_Meryem.pdf'
    file_path = r'C:\Users\claralado\Downloads\cv_clara_(3).pdf'
    client = chromadb.HttpClient()
    collection = client.create_collection("sample_collection")

    
    # Cargar y trocear el PDF
    pdf_text, pdf_images = extract_text_and_images(file_path)
    pdf_chunks = chunk_text(pdf_text)
    
    print(type(pdf_chunks))

    # Paso 2: Generar embeddings para los chunks
    embeddings = generate_embeddings(pdf_chunks)
    

    