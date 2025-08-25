import os
import bcrypt
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DB_PATH = "data/conversations.db"
os.makedirs("data",exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}")
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    
class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, nullable=False)
    role = Column(String, nullable=False)
    content =  Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
Base.metadata.create_all(bind=engine)

# --------- User Management -------------------

def create_user(username: str, password: str)-> bool:
     
    """Create new user with hashed password. Returns False if user exists"""
    session = SessionLocal()
    existing = session.query(User).filter_by(username=username).first()
    if existing:
        session.close()
        return False
    
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    user = User(username=username, password_hash = hashed.decode("utf-8"))
    session.add(user)
    session.commit()
    session.close()
    return True

def authenticate_user(username:str, password:str)-> bool:
    '''Check if username + password  match '''
    session = SessionLocal()
    user = session.query(User).filter_by(username=username).first()
    session.close()
    if not user:
        return False
    return bcrypt.checkpw(password.encode("utf-8"), user.password_hash.encode("utf-8"))
    
# ------------------ Conversation Management -----------------
def save_message(username:str, role:str, content:str):
    session = SessionLocal()
    msg = Conversation(username=username, role=role, content=content)
    session.add(msg)
    session.commit()
    session.close()

def load_history(username:str):
    session = SessionLocal()
    messages = (
        session.query(Conversation)
        .filter_by(username=username)
        .order_by(Conversation.timestamp)
        .all()
    )
    
    return [(m.role, m.content) for m in messages]




    
    
    

