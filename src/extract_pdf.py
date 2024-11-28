import fitz
#import os
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
    return embeddings

def store_embeddings_in_qdrant(chunks, embeddings, qdrant_client, collection_name="pdf_chunks"):
    ''' Almacena los embeddings en Qdrant '''
    # Crear colección si no existe
    if not qdrant_client.get_collection(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vector_size=len(embeddings[0]),  # Dimensión del vector
            distance="Cosine"  # Distancia para búsquedas (Cosine, Euclidean, etc.)
        )
    
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






# Yo creo que el siguiente codigo, por ahora, no aporta demasiado
'''
def get_last_chunk_index(index_file): 
    if os.path.exists(index_file):
        with open(index_file, 'r') as file:
            return int(file.read().strip())     
    return -1

def update_last_chunk_index(index_file, index):
    with open(index_file, 'w') as file:
        file.write(str(index))
'''

if __name__ == '__main__':
    # ruta del archivo PDF
    file_path = r'C:\Users\claralado\Downloads\exercicio3.pdf'
    index_file = 'last_chunk_index.txt'

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





    # Índice del último trozo mostrado
    #last_index = get_last_chunk_index(index_file)
    #print(last_index)

    # Mostrar todos los trozos
    #for i in range(last_index + 1, len(pdf_chunks)):
    #    print(f"Chunk {i + 1}:\n{pdf_chunks[i]}\n")

    # Actualizar el índice del último trozo mostrado
    #update_last_chunk_index(index_file, len(pdf_chunks) - 1)