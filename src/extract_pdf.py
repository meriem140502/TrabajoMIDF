from sentence_transformers import SentenceTransformer, util
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import fitz
import torch


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



# def get_relevant_context(question, model, qdrant_client, collection_name="pdf_chunks", top_k=3):
#     '''Busca los contextos más relevantes desde Qdrant basado en una consulta del usuario'''
    # Verificar si la colección en Qdrant no está vacía
    # collection_info = qdrant_client.get_collection(collection_name=collection_name)
    # if collection_info.status != "green" or collection_info.points_count == 0:
    #     print("La colección en Qdrant está vacía o no es accesible.")
    #     return [], []

    # Generar embedding para el input del usuario
    # input_embedding = model.encode([question])

    # Asegurarse de que el vector de entrada está en el formato correcto (lista)
    #input_embedding = input_embedding.tolist()[0]

    # Realizar la búsqueda en Qdrant usando el embedding generado
    # search_result = qdrant_client.search(
    #     collection_name=collection_name,
    #     query_vector=input_embedding,  # Se pasa como lista
    #     limit=top_k
    # )


    # if not search_result:
    #     print("No se encontraron resultados relevantes.")
    #     return [], []

    # Extraer los resultados relevantes
    # relevant_context = []
    # relevant_embeddings = []
    
    # Filtrar solo aquellos puntos que contienen un vector válido
    # for hit in search_result:
    #     if hit.vector is not None:
    #         relevant_context.append(hit.payload["chunk"])
    #         relevant_embeddings.append(hit.vector)
    
    # if not relevant_context:
    #     print("No se encontraron contextos relevantes con embeddings válidos.")
    #     return [], []

    # Calcular la similitud coseno explícita entre el input y los embeddings de Qdrant
    # cos_scores = [
    #     util.cos_sim(torch.tensor(input_embedding), torch.tensor(hit.vector)).item()
    #     for hit in search_result if hit.vector is not None
    # ]

    # Imprimir las similitudes calculadas
    # for idx, score in enumerate(cos_scores):
    #     print(f"Contexto {idx + 1}: {relevant_context[idx]} (Similitud coseno: {score})")

    # return relevant_context, relevant_embeddings


if __name__ == '__main__':
    # Ruta del archivo PDF
    #file_path = r'C:\Users\Usuario\Desktop\MIDF\TICs\TrabajoMIDF\src\CV_Meryem.pdf'
    file_path = r'C:\Users\ldebe\Downloads\CV.pdf'

    # Cargar y trocear el PDF
    pdf_text, pdf_images = extract_text_and_images(file_path)
    pdf_chunks = chunk_text(pdf_text)
    
    # print(pdf_chunks)

    # Cargar el modelo
    model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)

    # Paso 2: Generar embeddings para los chunks
    embeddings = generate_embeddings(pdf_chunks)

    # # Definir una pregunta
    #question = "Formación académica"

    
    # question = "¿Cual es la fecha de la graduación?"

    # # Buscar los contextos más relevantes en Qdrant
#     relevant_context, relevant_embeddings = get_relevant_context(
#     question=question,
#     model=model,  # El modelo de embeddings que cargaste antes
#     qdrant_client=qdrant_client,
#     collection_name="pdf_chunks",  # La colección que creaste
#     top_k=3  # Número de resultados relevantes que deseas obtener
# )

    
    # relevant_context, relevant_embeddings = get_relevant_context(question, model, qdrant_client)
    
    # print("\nContextos más relevantes encontrados:")
    # for idx, context in enumerate(relevant_context):
    #     print(f"Contexto {idx + 1}: {context}")
