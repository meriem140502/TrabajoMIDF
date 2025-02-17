# RAG AGAINST THE MACHINE
Este proyecto consiste en la implementación de un sistema de Retrieval-Augmented Generation (RAG) de código abierto que le permita formular preguntas concretas a un usuario en lenguaje natural sobre un fichero PDF o JSON (conversación de Telegram) y obtener una respuesta coherente teniendo en cuenta únicamente la información proporcionada. 

## Características 
  - **Procesamiento de PDF y JSON** para extraer información relevante. 
  - **Integración con modelos de lenguaje** (LLaMA 3.2 y DeepSeek) para respuestas coherentes. 
  - **Interfaz web interactiva** para facilitar la interacción del usuario.
  - **Uso de ChromaDB o Qdrant** como bases de datos vectoriales.

## Instalación
### **1. Clonar el repositorio**
```bash
git clone https://github.com/meriem140502/TrabajoMIDF.git cd rag-against-the-machine
```

### **2. Instalar dependencias**
Asegúrate de tener **Python 3.9+** instalado. 
- Instalación de la herramienta Ollama (https://ollama.com/download)
- Instalación Llama 3.2 
- Obtención de token para el uso en huggingface e instalación de DeepSeek
- Instalación de todas las librerías empleadas en los diferentes archivos de código
```bash
# Descargar el modelo LLaMA 3.2
ollama pull llama3.2

# Instalar DeepSeek (requiere token de Hugging Face)
pip install deepseek
```

## Utilización
Para poner en marcha el proyecto es necesario:
  1.	Ejecutar fichero ‘chroma & deepseek.py’ o bien ‘chroma & llama3.2.py’ o ‘qdrant & llama3.2.py’. 
´´´bash
  python "chroma & deepseek.py"
  python "chroma & llama3.2.py"
  python "qdrant & llama3.2.py"
´´´
  2.	Despliegue de interfaz web: abrir el navegador e ingresar la dirección IP y puerto correspondiente: 
  3.	Seleccionar modelo.
  4.	Cargar archivo pdf/json
  5.	Hacer pregunta
