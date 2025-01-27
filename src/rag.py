import json
import os
from flask import Flask, request, jsonify, render_template
import ollama
from sentence_transformers import SentenceTransformer
import chromadb
import fitz
import language_tool_python
from telegram import extraer_texto_por_id

app = Flask(__name__, template_folder='templates', static_folder='static')


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
    _, file_extension = os.path.splitext(file_path)

    if file_extension.lower() == '.pdf':
        doc = fitz.open(file_path)
        text = '\n'.join(page.get_text() for page in doc)
        tool = language_tool_python.LanguageTool('es')
        corrected_text = language_tool_python.utils.correct(text, tool.check(text))
        return corrected_text, False
    
    elif file_extension.lower() == '.json': 
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            corrected_text = extraer_texto_por_id(data)
            return corrected_text, True

    else:
        raise ValueError(f"El archivo con extensión {file_extension} no es compatible. Solo se aceptan .pdf y .json.")

def chunk_text(texto, is_json, chunk_size=100):
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

def generate_response(results, question, model_name='llama3.2'):
    """Genera una respuesta en lenguaje natural usando Ollama."""
    texts = [meta['chunk'] for meta in results['metadatas'][0]]
    prompt = f"Responde a la pregunta basándote ÚNICAMENTE en el siguiente texto:\n\n{' '.join(texts)}\n\nPregunta: {question}"
    client = ollama.Client(host="http://localhost:11434")
    try:
        result = client.chat(model=model_name, messages=[{'role': 'user', 'content': prompt}])
        return result['message']['content']
    except Exception as e:
        return f"Error al generar respuesta: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    file = request.files['file']
    file_path = os.path.join('./uploads', file.filename)
    os.makedirs('./uploads', exist_ok=True)
    file.save(file_path)
    
    text, is_json = extract_text(file_path)
    chunks = chunk_text(text, is_json)
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
    
    response = generate_response(results, question)
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
