import fitz
import os

def extract_text_and_images(file_path):
    doc = fitz.open(file_path)
    text = ''
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
        images.extend(page.get_images(full=True))
    return text, images

def chunk_text(text, chunk_size=75):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def get_last_chunk_index(index_file):
    if os.path.exists(index_file):
        with open(index_file, 'r') as file:
            return int(file.read().strip())
    return -1

def update_last_chunk_index(index_file, index):
    with open(index_file, 'w') as file:
        file.write(str(index))

# ruta del archivo PDF
file_path = r'C:\Users\ldebe\Downloads\CV.pdf'
index_file = 'last_chunk_index.txt'

# Cargar y trocear el PDF
pdf_text, pdf_images = extract_text_and_images(file_path)
pdf_chunks = chunk_text(pdf_text)

# Índice del último trozo mostrado
last_index = get_last_chunk_index(index_file)

# Mostrar el siguiente trozo
next_index = last_index + 1
if next_index < len(pdf_chunks):
    print(f"Chunk {next_index + 1}:\n{pdf_chunks[next_index]}\n")
    update_last_chunk_index(index_file, next_index)
else:
    print("No hay más trozos para mostrar.")
    update_last_chunk_index(index_file, -1)