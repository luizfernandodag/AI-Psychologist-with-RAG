import os
from time import sleep
import streamlit as st
from langchain_community.document_loaders import (WebBaseLoader,
                                                  YoutubeLoader, 
                                                  CSVLoader, 
                                                  PyPDFLoader, 
                                                  TextLoader)
from fake_useragent import UserAgent

def  load_site(url):
    ''''Load Content from a website'''
    document = ''
    for i in range(5):
        try:
            os.environ['USER_AGENT'] = UserAgent().random
            loader = WebBaseLoader(url, raise_for_status=True)
            docs = loader.load()
            document = '\n\n'.join([doc.page_content for doc in docs])
            break
        except:
            print(f'Error loading site attempt {i+1}')
            sleep(3)
    if document == '':
        st.error('Could not load the website.')
        st.stop()
    return document

def load_youtube(video_id):
    loader = YoutubeLoader(video_id, add_video_info=False, language=['pt'])
    docs = loader.load()
    document = '\n\n'.join([doc.page_content for doc in docs ])
    return document

def load_csv(caminho):
    loader = CSVLoader(caminho)
    docs = loader.load()
    document = '\n\n'.join([doc.page_content for doc in docs])
    return document

def load_pdf(caminho):
    loader = PyPDFLoader(caminho)
    docs = loader.load()
    document = '\n\n'.join([doc.page_content for doc in docs])
    return document

def load_txt(caminho):
    loader = TextLoader(caminho)
    docs = loader.load()
    document = '\n\n'.join([doc.page_content for doc in docs])
    return document
