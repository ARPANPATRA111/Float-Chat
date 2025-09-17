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

# In rag_system.py, replace only the RAGSQLQueryExecutor class

class RAGSQLQueryExecutor:
    def __init__(self, ollama_model=None):
        self.db_manager = DatabaseManager()
        self.vector_store = VectorStoreManager()
        try:
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

        self.sql_prompt_template = """You are an expert oceanographer and SQL developer for a PostGIS-enabled ARGO float database. Your task is to write a single, simple, and efficient PostgreSQL query to answer the user's question.

### RULES:
1.  **ALWAYS** use a simple `WHERE` clause for filtering. Do **NOT** use complex subqueries or `JOIN`s if a simple `WHERE` clause is sufficient.
2.  For geospatial "nearest" queries, you **MUST** use the `geom` column with the `<->` distance operator. The `geom` column is **ONLY** for location queries.
3.  For any filtering by date or time, you **MUST** use the `time` column.
4.  For complex analytical queries involving "anomaly" or "correlation", retrieve the relevant time-series data. Do not attempt to calculate these statistics directly in SQL.
5.  `ST_MakePoint` expects `(longitude, latitude)`.
6.  Return ONLY the SQL query. Do not add any explanation, markdown, or comments.
7.  The table name is `argo_profiles`.

### EXAMPLES:
User Question: How many unique floats are there?
SQL Query: SELECT COUNT(DISTINCT float_id) FROM argo_profiles;

User Question: Show me the temperature depth profile for float 5906256.
SQL Query: SELECT depth, temperature, float_id, profile_number FROM argo_profiles WHERE float_id = '5906256' ORDER BY depth ASC;

User Question: What are the 5 nearest floats to 15.29 N, 73.91 E?
SQL Query: SELECT float_id, lat, lon FROM argo_profiles ORDER BY geom <-> ST_SetSRID(ST_MakePoint(73.91, 15.29), 4326) LIMIT 5;
---

### DATABASE SCHEMA:
- Table: `argo_profiles`
- Columns: `float_id`, `time` (timestamp), `lat`, `lon`, `depth`, `temperature`, `salinity`, `geom` (geospatial point), `profile_number`

### CONTEXT FROM DATA METADATA:
{context}

### CURRENT USER QUESTION:
{question}

SQL Query:"""

        # --- NEW: A much better response prompt ---
        self.response_prompt_template = """You are an expert oceanographer acting as a helpful AI assistant. The user is seeing a chart generated from the data below. Your task is to provide a brief, insightful summary that guides the user.

### User Question:
{question}

### Data Summary:
{data_summary}

### Your Response Should:
1.  Acknowledge that the requested data/chart is being displayed.
2.  Provide one or two clear, simple oceanographic insights based on the data summary (e.g., "The data shows the typical ocean pattern where temperature decreases as depth increases.").
3.  Keep the response friendly, concise, and helpful. Do not say "I cannot generate a plot."

Analysis:"""

    # --- The rest of the RAGSQLQueryExecutor class is the same as before ---
    # (No changes needed for _generate_sql, _summarize_results, or query_with_rag)
    def _generate_sql(self, question: str, context: str):
        prompt = self.sql_prompt_template.format(context=context, question=question)
        response = self.llm.invoke(prompt)
        sql_query = response.strip().replace("```sql", "").replace("```", "").replace(";", "") + ";"
        return sql_query
    
    def _summarize_results(self, question: str, df: pd.DataFrame):
        if df.empty:
            return "The query returned no results. This could mean there is no data matching your criteria."
        
        # Create a text summary for the LLM
        if len(df) == 1 and len(df.columns) == 1:
            value = df.iloc[0, 0]
            col_name = df.columns[0]
            if isinstance(value, float): value_str = f"{value:,.2f}"
            else: value_str = str(value)
            data_summary = f"The query returned a single value for '{col_name}': {value_str}."
        else:
            data_summary = f"The query returned {len(df)} data points. Here is a sample:\n{df.head().to_string()}"
            
        prompt = self.response_prompt_template.format(question=question, data_summary=data_summary)
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