import json
import os
from flask import Flask, request, jsonify, render_template
import ollama
from sentence_transformers import SentenceTransformer
import chromadb
import fitz
import language_tool_python
from telegram import extraer_texto_por_id
from datetime import date
import re
 
 
app = Flask(__name__, template_folder='templates', static_folder='static')
is_json = False # variable global
 
def connect_to_chroma():
    """Conecta con la base de datos ChromaDB."""
    return chromadb.PersistentClient(path="./chroma_db")
 

def create_or_get_collection(client, collection_name="pdf_chunks"):
    """Accede a la colección en ChromaDB."""
    return client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})
 

def clear_collection(collection):
    """Elimina todos los documentos previos en la colección."""
    collection.delete(ids=collection.get()["ids"])
 

def extract_text(file_path):
    """Extrae el texto de un PDF o JSON y lo corrige gramaticalmente."""
    global is_json
    _, file_extension = os.path.splitext(file_path)
   
    if file_extension.lower() == '.pdf':
        is_json=False
        doc = fitz.open(file_path)
        text = '\n'.join(page.get_text() for page in doc)
        tool = language_tool_python.LanguageTool('es')
        corrected_text = language_tool_python.utils.correct(text, tool.check(text))
        return corrected_text
   
    elif file_extension.lower() == '.json':
        is_json = True
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            corrected_text = extraer_texto_por_id(data)
            return corrected_text
 
    else:
        raise ValueError(f"El archivo con extensión {file_extension} no es compatible. Solo se aceptan .pdf y .json.")
 

def chunk_text(texto, chunk_size=100):
    """Divide el texto en fragmentos."""
    if is_json:
        return texto['text'].tolist()
    else:
        words = texto.split()
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
 

def generate_embeddings(chunks, model):
    """Genera embeddings para los fragmentos de texto."""
    return model.encode(chunks, show_progress_bar=True)
 

def store_embeddings_in_chroma(collection, embeddings, chunks):
    """Almacena los embeddings en ChromaDB."""
    clear_collection(collection)
    collection.upsert(
        documents=chunks,
        metadatas=[{"chunk": chunk} for chunk in chunks],
        embeddings=embeddings,
        ids=[str(i) for i in range(len(chunks))]
    )
 

def search_in_chroma(collection, query_embedding, top_k=7):
    """Busca los fragmentos más relevantes en ChromaDB."""
    return collection.query(query_embeddings=[query_embedding], n_results=top_k)
 

def adjust_tense(response):
    """ Ajusta el tiempo verbal de la respuesta según la fecha del evento. """
    match = re.search(r"(\d{1,2}) de (\w+) de (\d{4})", response)
    
    if match:
        day, month_text, year = match.groups()
        month_mapping = {
            "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
            "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
            "xaneiro": 1, "febreiro": 2, "marzo": 3, "abril": 4, "maio": 5, "xuño": 6,
            "xullo": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "decembro": 12
        }

        month = month_mapping.get(month_text.lower())
        event_date = date(int(year), month, int(day))

        if event_date < date.today():
            response = response.replace("se realizará", "se ha llevado a cabo")
    
    return response


def generate_response(results, question, model_name='llama3.2'):
    """Genera una respuesta en lenguaje natural usando Ollama."""
    texts = [meta['chunk'] for meta in results['metadatas'][0]]
    prompt = f"Responde de manera directa y concisa a la pregunta basándote ÚNICAMENTE en el siguiente texto:\n\n{' '.join(texts)}\n\nPregunta: {question}"
    prompt += "\n\nNo respondas con 'Lo siento', 'No tengo información' ni frases similares. Si hay varias fechas sobre el mismo evento, prioriza la más reciente."
    
    if is_json:  
        prompt += f"\n\nAsegúrate de dar prioridad a la información más reciente o relevante, especialmente para preguntas relacionadas con fechas o eventos recientes. Teniendo en cuenta que hoy es día: {str(date.today())}"
    
    client = ollama.Client(host="http://localhost:11434")
    try:
        result = client.chat(model=model_name, messages=[{'role': 'user', 'content': prompt}])
        response = result['message']['content']
        return adjust_tense(response)
    except Exception as e:
        return f"Error al generar respuesta: {e}"


@app.route('/')
def index():
    current_file = os.path.basename(__file__)
    return render_template('index.html', current_file=current_file)

 
@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    file = request.files['file']
    file_path = os.path.join('./uploads', file.filename)
    os.makedirs('./uploads', exist_ok=True)
    file.save(file_path)
   
    text = extract_text(file_path)
    chunks = chunk_text(text)
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = generate_embeddings(chunks, model)
   
    client = connect_to_chroma()
    collection = create_or_get_collection(client)
    store_embeddings_in_chroma(collection, embeddings, chunks)
   
    return jsonify({"message": "Archivo procesado correctamente"})
 

@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.json['question']
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    query_embedding = generate_embeddings([question], model)[0]
   
    client = connect_to_chroma()
    collection = create_or_get_collection(client)
    results = search_in_chroma(collection, query_embedding, top_k=7)
    print(results)
    response = generate_response(results, question, model_name='llama3.2')
   
    return jsonify({"response": response})
 
if __name__ == '__main__':
    app.run(debug=True)