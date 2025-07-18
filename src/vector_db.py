import os
import glob
import xml.etree.ElementTree as ET
from typing import List, Dict
import shutil

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


class KnowledgeBaseVectorizer:
    """
    Clase para cargar, procesar, dividir en chunks y vectorizar documentos de una base de conocimiento XML u otros formatos.

    Parámetros
    ----------
    base_path : str
        Ruta raíz donde buscar documentos estructurados en subcarpetas.
    extension : str
        Extensión de archivos a procesar (por ejemplo, 'xml' o 'oet').
    db_name : str
        Ruta al directorio donde se almacenará la base de datos vectorial.
    """

    def __init__(self, base_path: str, extension: str, db_name: str):
        self.base_path = base_path
        self.extension = extension
        self.db_name = db_name
        self.documents = []
        self.processed_strings = []
        self.chunks = []

    def load_documents(self):
        """
        Carga todos los documentos con la extensión indicada desde cada subcarpeta de base_path.

        Cada subcarpeta se considera un tipo de documento distinto, y se adjunta un metadata 'doc_type'.
        """
        all_paths = glob.glob(os.path.join(self.base_path, "*"))
        folders = [path for path in all_paths if os.path.isdir(path)]
        for folder in folders:
            doc_type = os.path.basename(folder)
            try:
                loader = DirectoryLoader(
                    folder,
                    glob=f"**/*.{self.extension}",
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"},
                )
                folder_docs = loader.load()
                for doc in folder_docs:
                    doc.metadata["doc_type"] = doc_type
                self.documents.extend(folder_docs)
                print(f"[+] {len(folder_docs)} documentos cargados desde {doc_type}")
            except Exception as e:
                print(f"[-] Error cargando {folder}: {e}")

    def _parse_xml_structure(self, xml_content: str) -> List[Dict[str, str]]:
        """
        Parsea un string XML y extrae una lista de diccionarios con claves 'path' y 'text'.

        Cada entrada reprensenta un nodo de texto en la jerarquía XML, donde 'path' es la ruta de etiquetas.
        """
        def walk(node, path=""):
            entries = []
            tag = node.tag.split("}")[-1]
            new_path = f"{path}/{tag}" if path else tag

            if node.text and node.text.strip():
                entries.append({"path": new_path, "text": node.text.strip()})

            for child in node:
                entries.extend(walk(child, new_path))

            return entries

        try:
            root = ET.fromstring(xml_content)
            return walk(root)
        except ET.ParseError as e:
            print("Error al parsear XML:", e)
            return []

    def _flatten_parsed(self, parsed_entries: List[Dict[str, str]], sep: str = ": ") -> str:
        """
        Convierte la lista de estructuras {'path', 'text'} a un único string plano.

        Cada línea resulta de unir path y text con el separador dado.
        """
        return "\n".join(f"{entry['path']}{sep}{entry['text']}" for entry in parsed_entries)

    def parse_documents(self):
        """
        Parsea todos los documentos cargados y guarda sus textos aplanados en processed_strings.
        """
        self.processed_strings = []
        for doc in self.documents:
            parsed = self._parse_xml_structure(doc.page_content)
            flat = self._flatten_parsed(parsed)
            self.processed_strings.append(flat)

    def split_chunks(self, chunk_size: int = 400, chunk_overlap: int = 100):
        """
        Divide processed_strings en fragmentos aptos para embeddings.

        Usa CharacterTextSplitter para limitar cada chunk a `chunk_size` caracteres
        y una superposición de `chunk_overlap`. Ajustable para mantener chunks token-safe.
        """
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunks = splitter.create_documents(self.processed_strings)

    def create_vector_db(self):
        """
        Crea y persiste una base de datos vectorial Chroma en `db_name`.

        - Elimina cualquier base existente en esa ruta.
        - Usa OpenAIEmbeddings con chunk_size tokens-safe.
        - Inserta todos los chunks de una vez y persiste.
        """
        if not self.chunks:
            print("No hay chunks para procesar.")
            return

        if os.path.exists(self.db_name):
            shutil.rmtree(self.db_name)
        os.makedirs(self.db_name, exist_ok=True)

        embeddings = OpenAIEmbeddings(chunk_size=400)
        vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings,
            persist_directory=self.db_name
        )

        print(f"✅ Vector DB creada con {len(self.chunks)} chunks en {self.db_name}")
