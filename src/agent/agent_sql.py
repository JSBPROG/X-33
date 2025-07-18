#Agente que activa el flujo, detecta cambio y que cambio, además le pasa la tabla completa al agente_archetype

import os
import glob
from dotenv import load_dotenv
import gradio as gr

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain




MODEL = "gpt-4o-mini"
db_name = "vector_db"

load_dotenv()
os.environ['API_KEY'] = os.getenv('API_KEY', 'your-key-if-not-using-env')




        