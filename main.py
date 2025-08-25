import tempfile
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from db import save_message, load_history, create_user, authenticate_user
from langchain.text_splitter import RecursiveCharacterTextSplitter

from loaders import *

#Valid file Type

VALID_FILE_TYPES = ['Website',"Youtube", "PDF", "CSV", "TXT"]

#Model Providers

MODEL_CONFIG = {
    'Groq':{
        'models':['llama-3.1-70b-versatile', 'gemma2-9b-it', 'mixtral-8x7b-32768'],
        'chat': ChatGroq
        
    },
        
    'OpenAI':
        {
            'models': ['gpt-4o-mini', 'gpt-4o', 'o1-preview', 'o1-mini'],
            'chat': ChatOpenAI
            
        }
        
}

MEMORY = ConversationBufferMemory()

def login_page():
    ''' Login+signup page with password.'''
    st.title("🔐 Login to AI Psicologist")
    
    if "username" not in  st.session_state:
        st.session_state["username"] = None
        
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password",type="password", key="login_pass")
        
        if st.button("Login"):
            if authenticate_user(login_user, login_pass):
                st.session_state['username'] =  login_user
                st.success(f"✅ Welcome {login_user}!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

                
            
    with tab2:
        new_user = st.text_input("New Username", key="signup_user")
        new_pass = st.text_input("New Password", type="password", key="signup_pass")
        if st.button("Sign Up"):
            if create_user(new_user, new_pass):
                 st.success("🎉 User created successfully! Please log in.")
            else:
                st.error("⚠️ Username already exists")
    


def load_model(provider, model, api_key, file_type, file):
    """Load document, embed it, and initialize chat model"""
    match file_type:
        case 'Website':
            document = load_site(file)
        case 'Youtube':
            document = load_youtube(file)
        case 'PDF':
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
                temp.write(file.read())
                temp_name = temp.name 
            document = load_pdf(temp_name)

            
        case 'CSV':
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
                temp.write(file.read())
                temp_name = temp.name 
            document = load_csv(temp_name)
        case 'TXT':
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
                temp.write(file.read())
                temp_name = temp.name 
            document = load_txt(temp_name)
        

    # Create embeddings + vectorstore
    
    text_spliterr = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap =200,
        separators=["\n\n","\n"," ", ""]
    )
    
    docs = text_spliterr.split_documents(documents=document)

    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = Chroma.from_texts(docs, embedding=embeddings, persist_directory=".chroma_db")

    # Store in session
    st.session_state['vectorstore'] = vectorstore
    chat  = MODEL_CONFIG[provider]['chat'](model=model, api_key=api_key)
    st.session_state['chat'] = chat
    
def chat_page():
    """Main chat interface"""
    st.header('🤖 Welcome to the Psychologist AI', divider=True)
    
    chat_model = st.session_state.get('chat')
    vectorstore = st.session_state.get('vectorstore')
    memory = st.session_state.get('memory',MEMORY)
    
    # document loaded?
    
    if not vectorstore:
        st.warning("⚠️ Please upload a document with patient-psychologist conversations in the sidebar to start the session.")
        return
    

    
    # IA initialized?
    
    if not chat_model:
        st.info("ℹ️ Please select a model and click **Initialize Psychologist AI** in the sidebar.")
        return
    
    
    #Render past Conversations
    for m in memory.buffer_as_messages:
        chat = st.chat_message(m.type)
        chat.markdown(m.content)
        
    #User input
    user_input = st.chat_input('Talk to the AI Psicologist')
    if user_input and chat_model and vectorstore:
        st.chat_message('human').markdown(user_input)
        
        #Retrieve relevant docs
        docs = vectorstore.similarity_search(user_input, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        
        
        # Create prompt
        prompt = f"""
        You are a professional psycologist AI.
        Use the following excerpts from past patient-therapist conversations as reference:
        
        {context}
        
        Patient says:{user_input}
        """
        
        # Get model response
        ai_message = st.chat_message('ai')
        response = ai_message.write_stream(chat_model.stream(prompt))
        
        #Store memory
        
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(response)
        st.session_state['memory'] = memory 
        
def sidebar():
    """Sidebar for uploads and models selection"""
    tabs = st.tabs(['Uploads Documents', 'Model Selection'])
    
    with tabs[0]:
        file_type = st.selectbox('Select file type', VALID_FILE_TYPES)
        file = None
        
        if file_type == "WebSite":
            file = st.text_input('Enter website URL')
        if file_type == "Youtube":
            file = st.text_input('Enter Youtube video URL')
        if file_type == "PDF":
            file = st.file_uploader('Upload PDF file', type=['.pdf'])
        if file_type == "CSV":
            file = st.file_uploader('Upload CSV file', type=['.csv'])
        if file_type == "TXT":
            file = st.file_uploader('Upload TXT file', type=['.txt'])
            
    with tabs[1]:
        provider = st.selectbox('Select model provider', MODEL_CONFIG.keys())
        model = st.selectbox('Select model', MODEL_CONFIG[provider]['models'])
        
        api_key = st.text_input(
            f'Enter API key for {provider}',
            value=st.session_state.get(f'api_key_{provider}')
        )
        
        st.session_state[f'api_key_{provider}'] = api_key
        
        if st.button('Initialize Psychologist AI', use_container_width=True):
            load_model(provider, model, api_key, file_type, file)
    
        
        
                
    
    

def main():
    if not st.session_state.get("username"):
        login_page()
    else:
        chat_page()
        with st.sidebar:
            sidebar()

if __name__  == "__main__":
    main()

