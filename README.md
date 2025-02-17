# RAG AGAINST THE MACHINE
Este proyecto consiste en la implementación de un sistema de Retrieval-Augmented Generation (RAG) de código abierto que le permita formular preguntas concretas a un usuario en lenguaje natural sobre un fichero PDF o JSON (conversación de Telegram) y obtener una respuesta coherente teniendo en cuenta únicamente la información proporcionada. 

## Características 
  - **Procesamiento de PDF y JSON** para extraer información relevante. 
  - **Integración con modelos de lenguaje** (Llama 3.2 y DeepSeek) para respuestas coherentes. 
  - **Interfaz web interactiva** para facilitar la interacción del usuario.
  - **Uso de ChromaDB o Qdrant** como bases de datos vectoriales.

## Instalación
### **1. Clonar el repositorio**
```bash
git clone https://github.com/meriem140502/TrabajoMIDF.git cd rag-against-the-machine
```

### **2. Instalar dependencias**
Asegúrate de tener **Python 3.9+** instalado. 
- Instalación de la herramienta Ollama (https://ollama.com/download).
- Instalación Llama 3.2 .
- Obtención de token para el uso en huggingface e instalación de DeepSeek.
- Instalación de todas las librerías empleadas en los diferentes archivos de código.
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
  2.	Despliegue de interfaz web: abrir el navegador e ingresar la dirección IP y puerto correspondiente: http://127.0.0.1:5000
  3.	Seleccionar modelo.
  4.	Cargar archivo pdf/json.
  5.	Realizar pregunta.

## Pasos para obtención de Access Token 
  1. Registrarse en Hugging Face
  2. Iniciar sesión en Hugging Face
  3. En la parte superior derecha, haz clic en tu foto de perfil y selecciona Settings.
  4. En el menú lateral, haz clic en Access Tokens.
  5. Presiona el botón New Token.
  6. Ponle un nombre al token (por ejemplo, MiProyectoAI).
  7. Selecciona el nivel de permisos:
      - Read (Lectura) si solo vas a hacer consultas a modelos.
      - Write (Escritura) si necesitas subir modelos o datasets.
      - Admin si necesitas permisos completos.
  8. Haz clic en Generate Token.
  9. Copia el token y guárdalo en un lugar seguro. No lo compartas con nadie.

