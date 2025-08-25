An AI-powered psychologist chat app built with Streamlit, LangChain, RAG (Retrieval-Augmented Generation), Groq and OpenAI models.
It supports authentication, persistent memory, and contextual answers based on uploaded conversations (PDF, TXT, CSV, YouTube, or Websites).

Features

🔐 Login & Signup with password (stored in SQLite/db.py).

📂 Upload and process documents (PDF, TXT, CSV, YouTube transcripts, or website content).

🧩 RAG-powered retrieval using ChromaDB + OpenAI embeddings.

🧠 Conversation memory (LangChain ConversationBufferMemory).

🤖 Multiple LLM providers:

Groq: llama-3.1-70b-versatile, gemma2-9b-it, mixtral-8x7b-32768.

OpenAI: gpt-4o-mini, gpt-4o, o1-preview, o1-mini.

💬 Streamlit chat interface with conversation history.

🔎 Similarity search (RAG) for retrieving relevant conversation excerpts.


🛠️ Tech Stack

Python 3.12+

Streamlit

LangChain

ChromaDB

Groq API

OpenAI API

📂 Project Structure
.
├── app.py              # Main Streamlit application
├── db.py               # User authentication + message storage
├── loaders.py          # Functions to load PDF, TXT, CSV, YouTube, Website
├── requirements.txt    # Python dependencies
├── .chroma_db/         # Local vectorstore persistence
└── README.md           # This file

⚡ Installation

Clone this repo

git clone https://github.com/your-username/ai-psychologist.git
cd ai-psychologist


Create a virtual environment

python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows


Install dependencies

pip install -r requirements.txt

🔑 API Keys Setup

You will need API keys for:

OpenAI → Get here

Groq → Get here

You can set them in the app sidebar when selecting a model.

▶️ Running the App
streamlit run app.py

📝 Usage Guide

Login or Sign Up

Enter username and password (stored locally in SQLite).

Upload documents (from sidebar)

Supported: PDF, TXT, CSV, YouTube link, Website URL.

Select a model provider (Groq or OpenAI)

Enter the API key.

Choose a model from the dropdown.

Click Initialize Psychologist AI.

Chat with the AI Psychologist

The AI retrieves relevant excerpts using RAG.

The model answers in context of past conversations + uploaded docs.

📊 RAG Workflow

User uploads conversation files.

Files are chunked with RecursiveCharacterTextSplitter.

Each chunk is embedded using OpenAI embeddings.

Chunks are stored in ChromaDB (local vectorstore).

On user query:

Retrieve top-k relevant chunks (similarity_search_with_score).

Inject them into the prompt context.

AI responds with reference to past conversations.

📦 requirements.txt
streamlit
langchain
langchain-groq
langchain-openai
langchain-community
chromadb
tiktoken
pandas
pypdf
yt-dlp
beautifulsoup4

🚀 Next Steps (Improvements)

🔄 Multi-document history per user (per-patient RAG).

🗂️ Database storage for embeddings instead of local .chroma_db/.

📡 Deploy on Streamlit Cloud / AWS / GCP.

🧪 Add unit tests for loaders and db.

🔒 Encrypt user passwords (bcrypt/argon2).

👨‍💻 Author

Developed by Luiz Gadêlha ✨
💡 Contributions welcome! Open an issue or PR 🚀