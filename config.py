# --- config.py ---
import os

# Database configuration
# In config.py
DATABASE_CONFIG = {
    "sqlite": {
        "url": "sqlite:///argo.db"
    },
    "postgresql": {
        # MODIFIED: Use the user and password you just created
        "url": f"postgresql://argo_user:111111@localhost:5432/argo_db",
    }
}

# Use PostgreSQL by default for full feature support (PostGIS)
# Set to True only for lightweight local testing without location queries
USE_SQLITE = False

# Ollama configuration
OLLAMA_CONFIG = {
    "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    "model": "llama3.2",  # MODIFIED: Change this to "llama3.2"
    "temperature": 0.1,
    "top_p": 0.9
}

# Vector database configuration
VECTOR_DB_CONFIG = {
    "path": "./chroma_db",
    "collection_name": "argo_metadata" # MODIFIED: Collection now stores metadata
}

# Data processing configuration
DATA_PROCESSING_CONFIG = {
    "data_dir": "./data",
    "max_files": 100, # Set to a reasonable number for the hackathon
    "chunk_size": 5000 # For bulk inserts
}

# Application settings
APP_CONFIG = {
    "host": "0.0.0.0",
    "port": 8501,
    "debug": True
}

# Get environment variables with fallback to config values
def get_db_url():
    if USE_SQLITE:
        return DATABASE_CONFIG["sqlite"]["url"]
    else:
        return os.environ.get("DATABASE_URL", DATABASE_CONFIG["postgresql"]["url"])

def get_ollama_model():
    return os.environ.get("OLLAMA_MODEL", OLLAMA_CONFIG["model"])