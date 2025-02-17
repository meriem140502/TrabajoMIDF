import os
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import chromadb
import fitz
import language_tool_python
from telegram import extraer_texto_por_id
import json
import requests
import re
from datetime import date

app = Flask(__name__, template_folder='templates', static_folder='static')

# Configuración de Hugging Face
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"  # Reemplaza con el modelo que desees
HUGGINGFACE_ACCESS_TOKEN = "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"

def extract_text(file_path):
    """Extrae el texto de un PDF y lo corrige gramaticalmente."""
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
            corrected_test = extraer_texto_por_id(data)
            bool = True
        return corrected_test, bool
    else:
        raise ValueError(f"El archivo con extensión {file_extension} no es compatible. Solo se aceptan .pdf y .json.")


def chunk_text(texto, bool, chunk_size=100):
    """Divide el texto en fragmentos de tamaño especificado."""
    if bool:
        return texto['text'].tolist()
    else:
        words = texto.split()
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def connect_to_chroma():
    """Conecta con la base de datos ChromaDB."""
    return chromadb.PersistentClient(path="./chroma_db")


def create_or_get_collection(client, collection_name="pdf_chunks"):
    """Crea o accede a una colección en ChromaDB."""
    return client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})


def generate_embeddings(chunks, model):
    """Genera embeddings para los fragmentos de texto."""
    return model.encode(chunks, show_progress_bar=True)


def store_embeddings_in_chroma(collection, embeddings, chunks):
    """Almacena los embeddings en ChromaDB."""
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
            response = response.replace("se realiza", "se realizó")
    
    return response


def generate_response_huggingface(results, question):
    """Genera una respuesta en lenguaje natural usando la API de Hugging Face."""
    texts = [meta['chunk'] for meta in results['metadatas'][0]]
    context = ' '.join(texts)

    prompt = f"Responde a la siguiente pregunta basándote únicamente en el texto proporcionado:\n\nTexto: {context}\n\nPregunta: {question}\nRespuesta:"

    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 100, 
            "temperature": 0.1,  
            "do_sample": True, 
        }
    }

    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
        response.raise_for_status()

        generated_text = response.json()[0]['generated_text']
        generated_text = adjust_tense(generated_text)

        if "Respuesta:" in generated_text or "</think>" in generated_text:
            generated_text = generated_text.replace("</think>", "").strip()
            answer = generated_text.split("Respuesta:")[-1].strip()
        else:
            answer = generated_text.strip()

        # Eliminar duplicados
        answer = ' '.join(dict.fromkeys(answer.split()))

        return answer
    except Exception as e:
        return f"Error al generar respuesta: {e}"



@app.route('/')
def index():
    current_file = os.path.basename(__file__)
    return render_template('index.html', current_file=current_file)

@app.route('/select_model', methods=['POST'])
def select_model():
    data = request.get_json()
    selected_model = data.get('model', '')

    if selected_model:
        return jsonify({"model": selected_model})
    else:
        return jsonify({"error": "No se proporcionó un modelo"}), 400


@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    file = request.files['file']
    file_path = os.path.join('./uploads', file.filename)
    os.makedirs('./uploads', exist_ok=True)
    file.save(file_path)
    
    text, bool = extract_text(file_path)
    chunks = chunk_text(text, bool)
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = generate_embeddings(chunks, model)
    collection = create_or_get_collection(connect_to_chroma())
    store_embeddings_in_chroma(collection, embeddings, chunks)
    
    return jsonify({"message": "PDF procesado correctamente"})


@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.json['question']
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    query_embedding = generate_embeddings([question], model)[0]
    collection = create_or_get_collection(connect_to_chroma())
    results = search_in_chroma(collection, query_embedding, top_k=7)
    response = generate_response_huggingface(results, question)
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)