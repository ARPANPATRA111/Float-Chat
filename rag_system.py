# --- rag_system.py ---
import logging
import pandas as pd
from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer
import chromadb
from database_manager import DatabaseManager
from config import OLLAMA_CONFIG, VECTOR_DB_CONFIG, get_ollama_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=VECTOR_DB_CONFIG["path"])
        self.collection = self.client.get_or_create_collection(
            name=VECTOR_DB_CONFIG["collection_name"]
        )
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Could not load SentenceTransformer model: {e}")
            self.embedding_model = None

    def add_document(self, doc_id: str, document: str, metadata: dict):
        if not self.embedding_model: return
        try:
            embedding = self.embedding_model.encode(document).tolist()
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata]
            )
        except Exception as e:
            logger.error(f"ChromaDB - Error adding document {doc_id}: {e}")

    def search(self, query: str, n_results: int = 3):
        if not self.embedding_model: return []
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            logger.error(f"ChromaDB - Error searching: {e}")
            return []

# In rag_system.py, replace the entire class with this one.

class RAGSQLQueryExecutor:
    def __init__(self, ollama_model=None):
        self.db_manager = DatabaseManager()
        self.vector_store = VectorStoreManager()
        try:
            # MODIFIED: Using the newer LangChain community package for Ollama
            # Make sure to run: pip install langchain-ollama
            from langchain_ollama.llms import OllamaLLM
            self.llm = OllamaLLM(
                base_url=OLLAMA_CONFIG["base_url"],
                model=ollama_model or get_ollama_model(),
                temperature=OLLAMA_CONFIG["temperature"]
            )
            logger.info(f"✅ Connected to Ollama model: {self.llm.model}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama: {e}")
            raise

        # --- MODIFIED: A much more robust prompt with rules and examples ---
# In rag_system.py, inside the RAGSQLQueryExecutor class __init__ method

# Replace the entire self.sql_prompt_template with this:
# In rag_system.py, inside the RAGSQLQueryExecutor class __init__ method

# Replace the entire self.sql_prompt_template with this:
        self.sql_prompt_template = """You are an expert oceanographer and SQL developer for a PostGIS-enabled ARGO float database. Your task is to write a single, simple, and efficient PostgreSQL query to answer the user's question.

### RULES:
1.  **ALWAYS** use a simple `WHERE` clause for filtering. Do **NOT** use complex subqueries or `JOIN`s if a simple `WHERE` clause is sufficient.
2.  For geospatial "nearest" queries, you **MUST** use the `geom` column with the `<->` distance operator. The `geom` column is **ONLY** for location queries.
3.  **For any filtering by date or time, you MUST use the `time` column.** Do **NOT** use the `geom` column for time-based queries.
4.  `ST_MakePoint` expects `(longitude, latitude)`.
5.  Return ONLY the SQL query. Do not add any explanation, markdown, or comments.
6.  The table name is `argo_profiles`.

### EXAMPLES:
User Question: How many unique floats are there?
SQL Query: SELECT COUNT(DISTINCT float_id) FROM argo_profiles;

User Question: What is the average salinity for float 5906256?
SQL Query: SELECT AVG(salinity) FROM argo_profiles WHERE float_id = '5906256';

User Question: What are the 5 nearest floats to 15.29 N, 73.91 E?
SQL Query: SELECT float_id, lat, lon FROM argo_profiles ORDER BY geom <-> ST_SetSRID(ST_MakePoint(73.91, 15.29), 4326) LIMIT 5;
---

### DATABASE SCHEMA:
- Table: `argo_profiles`
- Columns: `float_id`, `time` (timestamp), `lat`, `lon`, `depth`, `temperature`, `salinity`, `geom` (geospatial point)

### CONTEXT FROM DATA METADATA:
{context}

### CURRENT USER QUESTION:
{question}

SQL Query:"""

        self.response_prompt_template = """You are an expert oceanographer. Summarize the following data to answer the user's question. Be concise and clear.

### User Question:
{question}

### Data:
{data_summary}

### Analysis:"""

    def _generate_sql(self, question: str, context: str):
        prompt = self.sql_prompt_template.format(context=context, question=question)
        # Using the newer .invoke() method from LangChain
        response = self.llm.invoke(prompt)
        # The new langchain-ollama returns the content directly
        sql_query = response.strip().replace("```sql", "").replace("```", "").replace(";", "") + ";"
        return sql_query
    
    def _summarize_results(self, question: str, df: pd.DataFrame):
        if df.empty:
            return "The query returned no results. This could mean there is no data matching your criteria."
        
        # If only one value is returned, make the response more direct
        if len(df) == 1 and len(df.columns) == 1:
            value = df.iloc[0, 0]
            col_name = df.columns[0]
            # Nicely format numbers
            if isinstance(value, float):
                value_str = f"{value:,.2f}"
            else:
                value_str = str(value)
            summary = f"The result for '{col_name}' is **{value_str}**."
        else:
            summary = df.head().to_string()
            if len(df) > 5:
                summary += f"\n... and {len(df) - 5} more rows."
            
        prompt = self.response_prompt_template.format(question=question, data_summary=summary)
        return self.llm.invoke(prompt)

    def query_with_rag(self, user_question: str):
        try:
            logger.info(f"Searching for context related to: '{user_question}'")
            context_docs = self.vector_store.search(user_question)
            context = "\n".join(f"- {doc}" for doc in context_docs) if context_docs else "No specific context found."
            
            logger.info(f"Generating SQL with context:\n{context}")
            sql_query = self._generate_sql(user_question, context)
            logger.info(f"Generated SQL: {sql_query}")

            query_result = self.db_manager.execute_query(sql_query)
            
            if not query_result["success"]:
                raise Exception(query_result["error"])

            df = pd.DataFrame(query_result['data'])
            enhanced_response = self._summarize_results(user_question, df)
            
            return {
                "success": True,
                "enhanced_response": enhanced_response,
                "generated_query": sql_query,
                "data": query_result['data'],
                "columns": query_result['columns']
            }
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "enhanced_response": f"I'm sorry, an error occurred: {e}",
                "data": []
            }