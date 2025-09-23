# --- Fixed rag_system.py ---
import logging
import pandas as pd
from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer
import chromadb
from database_manager import DatabaseManager
from config import OLLAMA_CONFIG, VECTOR_DB_CONFIG, get_ollama_model
import re

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

class QueryValidator:
    """Enhanced validator with better ocean region recognition"""
    
    def __init__(self):
        # Enhanced ocean/marine science related keywords
        self.ocean_keywords = {
            # Basic oceanographic terms
            'temperature', 'temp', 'salinity', 'sal', 'depth', 'pressure', 'ocean', 'sea', 'marine',
            'water', 'float', 'profile', 'argo', 'oceanographic', 'oceanography', 'floats',
            
            # Measurements and parameters
            'chlorophyll', 'chla', 'oxygen', 'doxy', 'ph', 'density', 'conductivity',
            'nitrate', 'phosphate', 'silicate', 'turbidity', 'fluorescence',
            
            # Ocean regions and geography - ENHANCED
            'latitude', 'longitude', 'lat', 'lon', 'location', 'position', 'coordinate',
            'north', 'south', 'east', 'west', 'atlantic', 'pacific', 'indian', 'arctic',
            'mediterranean', 'caribbean', 'gulf', 'bay', 'strait', 'channel',
            'arabian', 'bengal', 'red', 'black', 'caspian', 'barents', 'norwegian',
            'region', 'area', 'zone', 'basin', 'ridge', 'trench', 'plateau',
            
            # Countries and places near oceans - ADDED
            'india', 'mumbai', 'goa', 'chennai', 'kochi', 'australia', 'japan',
            'california', 'florida', 'norway', 'iceland', 'madagascar', 'maldives',
            
            # Time-related
            'time', 'date', 'year', 'month', 'day', 'season', 'seasonal', 'temporal',
            'recent', 'latest', 'historical', 'trend', 'series', 'current', 'present',
            
            # Analysis terms
            'average', 'mean', 'maximum', 'minimum', 'max', 'min', 'range', 'variation',
            'anomaly', 'correlation', 'pattern', 'distribution', 'profile', 'gradient',
            'all', 'show', 'display', 'list', 'find', 'search', 'get', 'what', 'where',
            
            # Spatial terms
            'near', 'nearest', 'around', 'within', 'between', 'above', 'below',
            'surface', 'deep', 'bottom', 'shallow', 'present', 'available', 'existing'
        }
        
        # Enhanced patterns for better recognition
        self.query_patterns = [
            r'\b(?:show|display|plot|graph|chart|list|find|get)\b',
            r'\b(?:what|where|when|how|which|all|present)\b',
            r'\b(?:temperature|salinity|depth|pressure|float|floats)\b',
            r'\b(?:ocean|sea|marine|water)\s+(?:region|area|basin)\b',
            r'\b(?:indian|pacific|atlantic|arctic)\s+ocean\b',
            r'\b(?:arabian|bengal|red|black)\s+sea\b',
            r'\bfloats?\s+(?:in|near|around|present)\b',
            r'\d+\s*(?:degrees?|°)\s*[ns]',  # Latitude patterns
            r'\d+\s*(?:degrees?|°)\s*[ew]',  # Longitude patterns
            r'\d+\s*(?:m|meters?|km|kilometers?)',  # Distance patterns
        ]
    
    def is_ocean_related(self, query: str) -> tuple[bool, float]:
        """Enhanced validation with better ocean region recognition"""
        query_lower = query.lower().strip()
        
        # Remove common punctuation and normalize
        normalized_query = re.sub(r'[^\w\s°]', ' ', query_lower)
        
        # Check for meaningless input (repeated characters, gibberish)
        if self._is_gibberish(normalized_query):
            logger.info(f"Query detected as gibberish: '{query[:30]}...'")
            return False, 0.0
        
        # Check for ocean-related keywords
        words = set(normalized_query.split())
        keyword_matches = len(words.intersection(self.ocean_keywords))
        keyword_score = min(keyword_matches / max(len(words), 1), 1.0)
        
        # Check for query patterns with higher weight
        pattern_matches = sum(1 for pattern in self.query_patterns 
                            if re.search(pattern, query_lower, re.IGNORECASE))
        pattern_score = min(pattern_matches / 2, 1.0)  # More lenient
        
        # Special boost for ocean regions
        ocean_region_boost = 0.0
        ocean_regions = ['indian ocean', 'pacific ocean', 'atlantic ocean', 'arctic ocean',
                        'arabian sea', 'bay of bengal', 'red sea', 'mediterranean sea']
        if any(region in query_lower for region in ocean_regions):
            ocean_region_boost = 0.5
        
        # Combined confidence score with region boost
        confidence = (keyword_score * 0.6) + (pattern_score * 0.4) + ocean_region_boost
        
        # More lenient threshold for ocean-related queries
        is_relevant = confidence > 0.05 or ocean_region_boost > 0
        
        logger.info(f"Query validation - '{query[:50]}...' -> Relevant: {is_relevant}, Confidence: {confidence:.3f}")
        
        return is_relevant, confidence
    
    def _is_gibberish(self, query: str) -> bool:
        """Detect gibberish but be less aggressive"""
        words = query.split()
        
        if not words:
            return True
        
        # Only flag truly meaningless patterns
        if len(words) >= 3:
            # Check if all words are identical (like "what is the what is the")
            if len(set(words)) <= 2 and all(len(w) <= 4 for w in words):
                return True
        
        # Check for single repeated character patterns
        if len(query.replace(' ', '')) <= 10 and len(set(query.replace(' ', ''))) <= 2:
            return True
        
        return False

class RAGSQLQueryExecutor:
    def __init__(self, ollama_model=None):
        self.db_manager = DatabaseManager()
        self.vector_store = VectorStoreManager()
        self.query_validator = QueryValidator()
        
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

        # Improved fallback responses
        self.fallback_responses = [
            "I'm designed to help you explore and analyze ARGO oceanographic data. Please ask questions about ocean temperature, salinity, depth profiles, float locations, or other marine science topics.",
            
            "I can help you with oceanographic data queries such as:\n• Temperature and salinity profiles\n• Float locations and movements\n• Depth analysis\n• Time series data\n• Geospatial ocean data\n\nPlease ask a question related to ocean or marine science data.",
            
            "I specialize in ARGO ocean float data analysis. Try asking about:\n• 'Show me floats in the Indian Ocean'\n• 'What are the nearest floats to Mumbai?'\n• 'Plot temperature vs depth profiles'\n• 'Find recent temperature data'"
        ]

        # FIXED SQL prompt template with better examples and error handling
        self.sql_prompt_template = """You are an expert oceanographer and SQL developer for a PostGIS-enabled ARGO float database. Generate a simple PostgreSQL query.

### CRITICAL RULES:
1. Use simple WHERE clauses, avoid complex subqueries
2. For spatial queries, use ST_Distance with geography casting for accuracy
3. Always use LIMIT to prevent huge results
4. The table name is ALWAYS 'argo_profiles'
5. Return ONLY the SQL query, no explanations

### COLUMN REFERENCE:
- float_id (text): Unique float identifier
- time (timestamp): Measurement timestamp  
- lat, lon (float): Coordinates
- depth (float): Depth in meters
- temperature, salinity (float): Measured values
- geom (geometry): PostGIS point geometry

### CORRECTED EXAMPLES:

User: "Show all floats in the Indian Ocean"
SQL: SELECT DISTINCT float_id, lat, lon FROM argo_profiles WHERE lat BETWEEN -60 AND 30 AND lon BETWEEN 20 AND 147 LIMIT 50;

User: "Find nearest floats to Mumbai (19.07°N, 72.87°E)"
SQL: SELECT float_id, lat, lon, ST_Distance(ST_MakePoint(lon, lat)::geography, ST_MakePoint(72.87, 19.07)::geography) as distance_m FROM argo_profiles ORDER BY ST_MakePoint(lon, lat)::geography <-> ST_MakePoint(72.87, 19.07)::geography LIMIT 10;

User: "Show temperature profiles for float 1234" 
SQL: SELECT depth, temperature, time FROM argo_profiles WHERE float_id = '1234' ORDER BY depth LIMIT 500;

User: "What floats are active recently?"
SQL: SELECT DISTINCT float_id, MAX(time) as latest_time FROM argo_profiles WHERE time > NOW() - INTERVAL '30 days' GROUP BY float_id ORDER BY latest_time DESC LIMIT 20;

### CONTEXT FROM DATA:
{context}

### USER QUESTION: 
{question}

SQL Query:"""

        self.response_prompt_template = """You are an expert oceanographer. The user asked about ocean data and received results. Provide a brief, helpful response.

### User Question:
{question}

### Data Summary:
{data_summary}

### Instructions:
1. Acknowledge what data is shown
2. Provide 1-2 simple oceanographic insights
3. Be encouraging and helpful
4. Keep response concise (2-3 sentences max)

Response:"""

    def _get_fallback_response(self) -> str:
        """Return a helpful fallback response"""
        import random
        return random.choice(self.fallback_responses)

    def _generate_sql(self, question: str, context: str):
        """Enhanced SQL generation with better error handling"""
        prompt = self.sql_prompt_template.format(context=context, question=question)
        
        try:
            response = self.llm.invoke(prompt)
            # Clean up the response
            sql_query = response.strip()
            # Remove any markdown formatting
            sql_query = re.sub(r'```sql\s*', '', sql_query)
            sql_query = re.sub(r'```\s*', '', sql_query)
            # Ensure it ends with semicolon
            if not sql_query.endswith(';'):
                sql_query += ';'
            
            logger.info(f"Generated SQL query: {sql_query}")
            return sql_query
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            # Return a safe fallback query
            return "SELECT COUNT(*) as total_records FROM argo_profiles LIMIT 1;"
    
    def _summarize_results(self, question: str, df: pd.DataFrame):
        if df.empty:
            return "No data found matching your criteria. This could mean the area or timeframe you specified doesn't have available measurements, or the query parameters were too restrictive."
        
        # Create a concise data summary for the LLM
        if len(df) == 1 and len(df.columns) == 1:
            value = df.iloc[0, 0]
            col_name = df.columns[0]
            if isinstance(value, (int, float)): 
                value_str = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
            else: 
                value_str = str(value)
            data_summary = f"Single result: {col_name} = {value_str}"
        else:
            # More concise summary
            cols = ', '.join(df.columns[:4])  # Only first 4 columns
            data_summary = f"Found {len(df)} records with columns: {cols}. Sample values: {df.head(2).to_dict('records')}"
            
        prompt = self.response_prompt_template.format(question=question, data_summary=data_summary)
        
        try:
            return self.llm.invoke(prompt)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I found {len(df)} records matching your query. The data includes measurements from ARGO ocean floats."

    def query_with_rag(self, user_question: str):
        try:
            # Validate query relevance
            is_relevant, confidence = self.query_validator.is_ocean_related(user_question)
            
            if not is_relevant:
                logger.info(f"Query rejected as irrelevant: '{user_question}' (confidence: {confidence:.3f})")
                return {
                    "success": True,
                    "enhanced_response": self._get_fallback_response(),
                    "generated_query": None,
                    "data": [],
                    "columns": [],
                    "fallback_used": True
                }
            
            # Process relevant queries
            logger.info(f"Processing relevant query: '{user_question}' (confidence: {confidence:.3f})")
            
            # Get context from vector store
            context_docs = self.vector_store.search(user_question)
            context = "\n".join(f"- {doc}" for doc in context_docs) if context_docs else "No specific float context found."
            
            # Generate SQL
            sql_query = self._generate_sql(user_question, context)
            
            # Execute query with better error handling
            query_result = self.db_manager.execute_query(sql_query)
            
            if not query_result["success"]:
                logger.error(f"SQL execution failed: {query_result.get('error', 'Unknown error')}")
                return {
                    "success": False,
                    "error": f"Database query failed: {query_result.get('error', 'Unknown error')}",
                    "enhanced_response": "I encountered an error while querying the database. This might be due to a syntax issue or database connectivity problem. Please try rephrasing your question or contact support.",
                    "data": [],
                    "fallback_used": False
                }

            # Process results
            df = pd.DataFrame(query_result['data'])
            enhanced_response = self._summarize_results(user_question, df)
            
            return {
                "success": True,
                "enhanced_response": enhanced_response,
                "generated_query": sql_query,
                "data": query_result['data'],
                "columns": query_result['columns'],
                "fallback_used": False
            }
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "enhanced_response": f"An unexpected error occurred while processing your oceanographic data query. Error details: {str(e)}",
                "data": [],
                "fallback_used": False
            }