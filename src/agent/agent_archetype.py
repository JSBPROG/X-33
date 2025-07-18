import os
import sys
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_chroma import Chroma

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vector_db import KnowledgeBaseVectorizer

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))


MODEL_NAME = os.getenv("MODEL", "gpt-4")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHETYPES_PATH = os.path.join(BASE_DIR, "..", "knowledge_base")
TEMPLATES_PATH = os.path.join(BASE_DIR, "..", "knowledge_base")
ARCHETYPES_DB_PATH = os.path.join(BASE_DIR, "..", "database", "vector_db_arch")
TEMPLATES_DB_PATH = os.path.join(BASE_DIR, "..", "database", "vector_db_temp")


def setup_knowledge_base(base_path: str, extension: str, db_name: str) -> Chroma:
    """
    Inicializa, procesa y vectoriza una base de conocimiento.

    Esta función ejecuta el pipeline completo: carga documentos, los parsea,
    los divide en fragmentos y crea una base de datos vectorial Chroma persistente.

    Args:
        base_path: El directorio raíz de los archivos de conocimiento.
        extension: La extensión de archivo de los documentos (ej., 'xml', 'oet').
        db_name: La ruta del directorio para guardar la base de datos vectorial.

    Returns:
        Una instancia de almacén de vectores Chroma.
    """
    if os.path.exists(db_name):
        print(f"--- Loading existing Knowledge Base from: {db_name} ---")
        return Chroma(persist_directory=db_name, embedding_function=OpenAIEmbeddings())

    vectorizer = KnowledgeBaseVectorizer(
        base_path=base_path,
        extension=extension,
        db_name=db_name
    )
    print(f"\n--- Processing Knowledge Base: {os.path.basename(base_path)} ---")
    vectorizer.load_documents()
    vectorizer.parse_documents()
    vectorizer.split_chunks()
    vectorizer.create_vector_db()
    
    return Chroma(persist_directory=db_name, embedding_function=OpenAIEmbeddings())


class ArchetypeAgent:
    """
    Un agente que aprovecha un Modelo de Lenguaje Grande y una base de datos vectorial para inferir
    arquetipos y responder consultas relacionadas con la interoperabilidad clínica.
    """
    def __init__(self, model_name: str, retriever: VectorStoreRetriever):
        """
        Inicializa el agente.

        Args:
            model_name: El nombre del modelo de OpenAI a usar.
            retriever: El recuperador de almacén de vectores configurado para usar como contexto.
        """
        self.llm = self._initialize_llm(model_name)
        self.conversation_chain = self._create_conversation_chain(retriever)

    def _initialize_llm(self, model_name: str) -> ChatOpenAI:
        """Configura el modelo ChatOpenAI con un prompt de sistema específico."""
        system_prompt = """Eres un experto en interoperabilidad clínica y estándares openEHR. Tu función es ayudar a seleccionar arquetipos y templates adecuados para representar información médica estructurada, a partir de preguntas o descripciones en lenguaje natural.

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
        return ChatOpenAI(
            temperature=0.7,
            model=model_name
        )

    def _create_conversation_chain(self, retriever: VectorStoreRetriever) -> ConversationalRetrievalChain:
        """Construye la cadena de recuperación conversacional con memoria."""
        system_prompt = """Eres un experto en interoperabilidad clínica y estándares openEHR. Tu función es ayudar a seleccionar arquetipos y templates adecuados para representar información médica estructurada, a partir de preguntas o descripciones en lenguaje natural.

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

        prompt = PromptTemplate(
            template=system_prompt + "\n\n---\n\nContext:\n{context}\n\nQuestion:\n{question}",
            input_variables=["context", "question"],
        )

        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
        )

    def ask(self, query: str) -> str:
        """
        Formula una pregunta al agente y devuelve la respuesta.

        Args:
            query: La pregunta del usuario en lenguaje natural.

        Returns:
            La respuesta de la cadena conversacional.
        """
        result = self.conversation_chain.invoke({"question": query})
        return result.get("answer", "No answer could be generated.")


def main():
    """
    Función principal para configurar las bases de conocimiento, inicializar el agente y ejecutar una consulta.
    """
    arch_db = setup_knowledge_base(
        base_path=ARCHETYPES_PATH,
        extension="xml",
        db_name=ARCHETYPES_DB_PATH
    )
    temp_db = setup_knowledge_base(
        base_path=TEMPLATES_PATH,
        extension="oet",
        db_name=TEMPLATES_DB_PATH
    )

    retriever = temp_db.as_retriever(search_kwargs={"k": 2})
    
    agent = ArchetypeAgent(model_name=MODEL_NAME, retriever=retriever)

    query = "¿Qué arquetipo puedo utilizar para la presión sanguínea?"
    print(f"\nQuery: {query}")
    answer = agent.ask(query)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
