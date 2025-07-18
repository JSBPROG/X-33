"""
Agente que utiliza LLM + base de datos vectorial para inferir arquetipos y responder consultas.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import sys


# Añadir el directorio padre al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))


# Importación desde nivel superior
from vector_db import KnowledgeBaseVectorizer



# ----------------------------
# Cargar variables de entorno
# ----------------------------

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-3.5-turbo")  # Modelo por defecto si no está definido

# ----------------------------
# Inicializar modelo LLM
# ----------------------------
llm = ChatOpenAI(
    temperature=0.7,
    model=MODEL,
    model_kwargs={
        "system_message": """Eres un experto en interoperabilidad clínica y estándares openEHR. Tu función es ayudar a seleccionar arquetipos y templates adecuados para representar información médica estructurada, a partir de preguntas o descripciones en lenguaje natural.

Dispones de una base de conocimiento completa que incluye todos los arquetipos y templates oficiales de openEHR, previamente cargados y procesados.

Cuando un usuario realiza una consulta, debes identificar si está solicitando:
- Qué **arquetipos** usar (unidades semánticas reutilizables).
- Qué **templates** aplicar (combinaciones predefinidas de arquetipos para situaciones clínicas específicas).
- Información técnica sobre un elemento clínico.

Siempre proporciona respuestas concretas. Si es posible, devuelve:
- El nombre del arquetipo o template.
- Su finalidad clínica.
- Sus rutas o conceptos principales si es relevante.

Si el término buscado no existe exactamente, sugiere el más similar. No inventes arquetipos o templates que no estén en openEHR.

Ejemplos de consultas:
- “Quiero registrar presión arterial, ¿qué arquetipo debo usar?”
- “¿Qué template se usa para anamnesis pediátrica?”
- “¿Hay algún arquetipo para antecedentes familiares?”

Responde de forma concisa, clara y clínica.
"""
    }
)
#  ------------------------
# Cargar y vectorizar archetypes
# ----------------------------
arch_vect = KnowledgeBaseVectorizer(
    base_path="src/knowledge_base/archetypes_2025_07_16-20_36_00",
    extension="xml",
    db_name="src/agent/vector_db_arch"
)
arch_vect.load_documents()
arch_vect.parse_documents()
arch_vect.split_chunks()
arch_vect.create_vector_db()

# ----------------------------
# Cargar y vectorizar templates
# ----------------------------
temp_vect = KnowledgeBaseVectorizer(
    base_path="src/knowledge_base/templates_2025_07_16-20_36_55",
    extension="oet",
    db_name="src/agent/vector_db_temp"
)
temp_vect.load_documents()
temp_vect.parse_documents()
temp_vect.split_chunks()
temp_vect.create_vector_db()

# ----------------------------
# Crear retrievers desde la base de datos
# ----------------------------
retriever_arch = Chroma(persist_directory="src/agent/vector_db_arch", embedding_function=OpenAIEmbeddings()).as_retriever()
retriever_temp = Chroma(persist_directory="src/agent/vector_db_temp", embedding_function=OpenAIEmbeddings()).as_retriever()

# ----------------------------
# Seleccionar uno de los retrievers (puedes crear lógica para elegir dinámicamente)
# ----------------------------
retriever = retriever_arch  # ← puedes cambiar esto dinámicamente si lo deseas

# ----------------------------
# Crear memoria de conversación
# ----------------------------
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# ----------------------------
# Construir cadena de conversación
# ----------------------------
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# ----------------------------
# Ejemplo de consulta
# ----------------------------
query = "¿Qué arquetipo puedo utilizar para la presión sanguínea??"
result = conversation_chain.invoke({"question": query})
print(result["answer"])
