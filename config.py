import os
from langchain_groq import ChatGroq
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY not found. Please set it in the .env file.")
