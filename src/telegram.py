import pandas as pd
from googletrans import Translator
import re


# Leer el archivo JSON
#with open(ruta, 'r', encoding='utf-8') as archivo:
#    datos = json.load(archivo)


def limpiar_texto(texto):
    # Expresión regular para detectar emoticonos (caracteres Unicode en rangos específicos)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticonos
        "\U0001F300-\U0001F5FF"  # Símbolos y pictogramas
        "\U0001F680-\U0001F6FF"  # Transporte y símbolos
        "\U0001F1E0-\U0001F1FF"  # Banderas
        "\U00002500-\U00002BEF"  # Caracteres chinos y similares
        "\U00002702-\U000027B0"  # Símbolos adicionales
        "\U000024C2-\U0001F251"  # Otros pictogramas
        "]+",
        flags=re.UNICODE,
    )
    texto_sin_emojis = emoji_pattern.sub("", texto)
    texto_limpio = texto_sin_emojis.replace("\n", " ")

    return texto_limpio


# Función para extraer textos asociados al ID del mensaje
def extraer_texto_por_id(mensajes):
    mensajes = datos.get("messages", [])
    textos_por_id = []
    for mensaje in mensajes:
        if "id" in mensaje and "text" in mensaje:  
            contenido = mensaje["text"]
            texto_completo = "" 
            if isinstance(contenido, list):  # Si el texto es una lista
                for item in contenido:
                    if isinstance(item, dict) and "text" in item:  # Si es un diccionario con "text"
                        texto_completo += limpiar_texto(item["text"]) + " "
                    elif isinstance(item, str):  # Si es un string
                        texto_completo += limpiar_texto(item) + " "
            elif isinstance(contenido, str):  # Si es un string directamente
                texto_completo = limpiar_texto(contenido)
            textos_por_id.append({"text": texto_completo.strip(), "date": mensaje["date"]})
    df = pd.DataFrame(textos_por_id)
    df = df[df['text'].str.strip() != ""]  # Eliminar textos vacíos o solo espacios
    df['fecha'] = pd.to_datetime(df['date']).dt.strftime('%d-%m-%Y')
    df['text'] = "Publicado el día " + df['fecha'] + ": " + df['text']
    return pd.DataFrame(df)







'''
# Traduccion de mensajes, solo si no están en castellano
translator = Translator()

def traducir_googletrans(texto):
    deteccion = translator.detect(texto)  # Detectar idioma
    if deteccion != 'es':  # Solo traducir si no está en español
        return translator.translate(texto, src=deteccion, dest='es').text
    return texto  # Si ya está en español, devolver el texto original


#df['texto'] = df['text'].apply(traducir_googletrans)
'''
