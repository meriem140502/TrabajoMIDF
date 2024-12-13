from sentence_transformers import SentenceTransformer, util
import chromadb
from chromadb.config import Settings
import fitz
import torch
import ollama


def extract_text(file_path):
    '''Función encargada de extraer solo el texto de un PDF'''
    doc = fitz.open(file_path)
    text = ''  # Variable para almacenar el texto
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()  # Extraer el texto de cada página
    return text  # Solo devolver el texto



def chunk_text(text, chunk_size=30):
    '''División en chunks del texto'''
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


def connect_to_chroma():
    client = chromadb.Client()  # Usar el constructor sin necesidad de configuraciones antiguas
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
    # Imprimir los embeddings
    
    """for i, embedding in enumerate(embeddings):
        print(f"Embedding {i}: {embedding}")
    """
    return embeddings
    

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


def search_in_chroma(collection, query_embedding, top_k=1):
    """Buscar en Chroma los contextos más relevantes"""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results


def main(file_path, collection_name="pdf_chunks", top_k=3):
    # Paso 1: Cargar texto y dividir en chunks (ajustar según tu necesidad)
    pdf_text = extract_text(file_path)  # Suponiendo que ya tienes esta función
    pdf_chunks = chunk_text(pdf_text)  # Ajusta según la función que tengas para dividir el texto

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
    question = "¿que objetivo tiene una ley organica?"
    question_embedding = generate_embeddings([question], model)[0]  # Generar embedding de la consulta

    # Paso 6: Buscar contextos relevantes
    results = search_in_chroma(collection, question_embedding, top_k)

    # Mostrar los resultados
    for result in results['documents']:
        print(f"Contexto relevante: {result}")

# Llamar a la función principal con el archivo PDF
if __name__ == "__main__":
    file_path = r'C:\Users\ldebe\Downloads\Tema_Derecho.pdf'
    main(file_path)

# if __name__ == '__main__':
#     # Ruta del archivo PDF
#     file_path = r'C:\Users\Usuario\Desktop\MIDF\TICs\TrabajoMIDF\src\CV_Meryem.pdf'
    
#     # Cargar y trocear el PDF
#     pdf_text, pdf_images = extract_text_and_images(file_path)
#     pdf_chunks = chunk_text(pdf_text)
    
#     # print(pdf_chunks)

#     # Cargar el modelo
#     model_name = 'all-MiniLM-L6-v2'
#     model = SentenceTransformer(model_name)

#     # Paso 2: Generar embeddings para los chunks
#     embeddings = generate_embeddings(pdf_chunks)
    

#     # Paso 3: Conectar a Qdrant
#     qdrant_client = QdrantClient(":memory:")  # Usa Qdrant en memoria (para pruebas locales)
#     # print(qdrant_client.get_collections())
    


#     # Usa `QdrantClient(host="localhost", port=6333)` si tienes un servidor Qdrant corriendo

#     # Paso 4: Almacenar embeddings en Qdrant
#     store_embeddings_in_qdrant(pdf_chunks, embeddings, qdrant_client)
    
#     # colecciones = qdrant_client.get_collections()
#     # print("colecciones =" ,colecciones)
    
#     # Verificar el contenido de la colección
#     collection_info = qdrant_client.get_collection(collection_name="pdf_chunks")
#     print(f"Información de la colección: {collection_info}")

#     # Listar los puntos almacenados
#     points = qdrant_client.scroll(collection_name="pdf_chunks", limit=10)
#     print("Puntos en la colección:", points)

#     # print(f"Embeddings generados y almacenados en Qdrant. Total chunks: {len(pdf_chunks)}")

#     # # Definir una pregunta
#     question = "Formación académica"

    
#     # question = "¿Cual es la fecha de la graduación?"

#     # # Buscar los contextos más relevantes en Qdrant
#     relevant_context, relevant_embeddings = get_relevant_context(
#     question=question,
#     model=model,  # El modelo de embeddings que cargaste antes
#     qdrant_client=qdrant_client,
#     collection_name="pdf_chunks",  # La colección que creaste
#     top_k=3  # Número de resultados relevantes que deseas obtener
# )

    
#     # relevant_context, relevant_embeddings = get_relevant_context(question, model, qdrant_client)
    
#     # print("\nContextos más relevantes encontrados:")
#     # for idx, context in enumerate(relevant_context):
#     #     print(f"Contexto {idx + 1}: {context}")
