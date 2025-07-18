import os
import glob
import xml.etree.ElementTree as ET
from typing import List, Dict

from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


class KnowledgeBaseVectorizer:
    """
    Clase para cargar, procesar, chunkear y vectorizar documentos de una base de conocimiento XML/OET.
    
    Parameters
    ----------
    base_path : str
        Ruta raíz donde buscar documentos.
    extension : str
        Extensión de archivos a procesar (por ejemplo, 'xml' o 'oet').
    db_name : str
        Nombre del directorio donde se almacenará la base de datos vectorial.
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
        Carga documentos desde carpetas contenidas en la ruta `base_path`.

        Cada subcarpeta se trata como un tipo de documento distinto. Se cargan todos
        los archivos con la extensión indicada y se agregan metadatos del tipo de documento.
        
        Returns
        -------
        None
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
        Parsea el contenido XML y devuelve una lista de rutas jerárquicas con texto asociado.

        Parameters
        ----------
        xml_content : str
            Contenido plano del archivo XML.

        Returns
        -------
        List[Dict[str, str]]
            Lista de diccionarios con las claves 'path' y 'text' representando cada nodo de texto en el XML.
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
        Convierte la estructura jerárquica en una cadena de texto plano con rutas y textos.

        Parameters
        ----------
        parsed_entries : List[Dict[str, str]]
            Lista de rutas y textos extraídos del XML.
        sep : str, optional
            Separador entre la ruta y el texto. Por defecto es ": ".

        Returns
        -------
        str
            Texto plano resultante.
        """
        return "\n".join(f"{entry['path']}{sep}{entry['text']}" for entry in parsed_entries)

    def parse_documents(self):
        """
        Parsea todos los documentos cargados y los convierte a texto estructurado plano.

        Returns
        -------
        None
        """
        self.processed_strings = []
        for doc in self.documents:
            parsed = self._parse_xml_structure(doc.page_content)
            flat = self._flatten_parsed(parsed)
            self.processed_strings.append(flat)

    def split_chunks(self, chunk_size=1000, chunk_overlap=200):
        """
        Divide los textos procesados en fragmentos (chunks) para su vectorización.

        Parameters
        ----------
        chunk_size : int, optional
            Longitud máxima de cada fragmento. Por defecto es 1000 caracteres.
        chunk_overlap : int, optional
            Superposición entre fragmentos consecutivos. Por defecto es 200 caracteres.

        Returns
        -------
        None
        """
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunks = splitter.create_documents(self.processed_strings)

    def create_vector_db(self):
        """
        Crea y persiste la base de datos vectorial usando Chroma y OpenAIEmbeddings.

        Si ya existe una base con el mismo nombre, se sobrescribe.

        Returns
        -------
        None
        """
        embeddings = OpenAIEmbeddings()

        if os.path.exists(self.db_name):
            Chroma(persist_directory=self.db_name, embedding_function=embeddings).delete_collection()

        vectorstore = Chroma.from_documents(documents=self.chunks, embedding=embeddings, persist_directory=self.db_name)
        print(f"🧠 Vector DB creada con {vectorstore._collection.count()} chunks en {self.db_name}")



