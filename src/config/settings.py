import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Neo4j database settings
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
AURA_INSTANCEID = os.getenv("AURA_INSTANCEID")
AURA_INSTANCENAME = os.getenv("AURA_INSTANCENAME")

# LLM model settings
OPENAI_LLM_MODEL = "gpt-4o-2024-08-06"
DEFAULT_LLM_MODEL = "llama3-70b-8192"
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"
GROQ_API_URL = "https://api.groq.com/openai/v1"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Text processing settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 24

# Graph settings
BASE_ENTITY_LABEL = True
INCLUDE_SOURCE = True