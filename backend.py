import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

load_dotenv()

if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please make sure it's set correctly.")

class SQLQueryExecutor:
    def __init__(self, db_path: str = "sqlite:///argo.db"):
        self.db_path = db_path
        self.sqlite_path = db_path.replace("sqlite:///", "")
        
        # Initialize LangChain components
        self.db = SQLDatabase.from_uri(db_path)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0,
            google_api_key=os.environ.get("GEMINI_API_KEY")
        )
        self.query_chain = create_sql_query_chain(self.llm, self.db)
    
    def generate_sql_query(self, user_question: str) -> str:
        """
        Generate SQL query from natural language question using Gemini.
        """
        try:
            # The LLM's response is a verbose string
            llm_response = self.query_chain.invoke({"question": user_question})

            # ✅ FIX: Parse the response to extract only the SQL query
            if "SQLQuery:" in llm_response:
                sql_query = llm_response.split("SQLQuery:")[1].strip()
            else:
                # Fallback for other formats
                sql_query = llm_response.strip()

            # Clean up markdown formatting if present (good practice to keep)
            if sql_query.startswith("```sql"):
                sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            elif sql_query.startswith("```"):
                sql_query = sql_query.replace("```", "").strip()
            
            return sql_query
        except Exception as e:
            raise Exception(f"Error generating SQL query: {str(e)}")
    
    def execute_query(self, sql_query: str, max_rows: int = 100) -> Dict[str, Any]:
        """
        Execute SQL query and return results as a dictionary with metadata.
        """
        try:
            # Connect to SQLite database
            conn = sqlite3.connect(self.sqlite_path)
            
            # Execute query and get results
            df = pd.read_sql_query(sql_query, conn)
            
            # Close connection
            conn.close()
            
            # Limit results if needed
            if len(df) > max_rows:
                df_limited = df.head(max_rows)
                truncated = True
            else:
                df_limited = df
                truncated = False
            
            # Prepare result dictionary
            result = {
                "success": True,
                "data": df_limited.to_dict('records'),  # Convert to list of dictionaries
                "columns": df_limited.columns.tolist(),
                "total_rows": len(df),
                "displayed_rows": len(df_limited),
                "truncated": truncated,
                "query": sql_query
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": sql_query,
                "data": [],
                "columns": [],
                "total_rows": 0,
                "displayed_rows": 0,
                "truncated": False
            }
    
    def query_and_execute(self, user_question: str, max_rows: int = 100) -> Dict[str, Any]:
        """
        Complete pipeline: generate SQL from natural language and execute it.
        """
        try:
            # Step 1: Generate SQL query
            sql_query = self.generate_sql_query(user_question)
            print(f"Generated SQL Query:\n{sql_query}\n")
            
            # Step 2: Execute the query
            result = self.execute_query(sql_query, max_rows)
            result["original_question"] = user_question
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "original_question": user_question,
                "query": "",
                "data": [],
                "columns": [],
                "total_rows": 0,
                "displayed_rows": 0,
                "truncated": False
            }
    
    def display_results(self, result: Dict[str, Any]) -> None:
        """
        Display query results in a formatted way.
        """
        print(f"📝 Question: {result.get('original_question', 'N/A')}")
        print(f"🔍 Generated SQL: {result.get('query', 'N/A')}")
        print(f"✅ Success: {result['success']}")
        
        if result['success']:
            print(f"📊 Total rows: {result['total_rows']}")
            print(f"👀 Displayed rows: {result['displayed_rows']}")
            
            if result['truncated']:
                print(f"⚠️  Results truncated (showing first {result['displayed_rows']} of {result['total_rows']} rows)")
            
            if result['data']:
                # Convert back to DataFrame for pretty printing
                df = pd.DataFrame(result['data'])
                print("\n📋 Results:")
                print(df.to_string(index=False))
            else:
                print("📋 No data returned")
        else:
            print(f"❌ Error: {result['error']}")
        
        print("-" * 80)

# Convenience functions for backward compatibility
def get_sql_query(user_question: str) -> str:
    """
    Legacy function - just generates SQL query (for backward compatibility).
    """
    executor = SQLQueryExecutor()
    return executor.generate_sql_query(user_question)

def query_database(user_question: str, max_rows: int = 100) -> Dict[str, Any]:
    """
    New function - generates and executes SQL query, returns results.
    """
    executor = SQLQueryExecutor()
    return executor.query_and_execute(user_question, max_rows)

# ==========================
# Test the enhanced backend
# ==========================
if __name__ == "__main__":
    print("🤖 Enhanced Text-to-SQL Backend is ready (using Gemini ✨)")
    print("Now with query execution capabilities!\n")
    
    # Initialize executor
    executor = SQLQueryExecutor()
    
    # Example questions to test
    test_questions = [
        "How many unique floats are in the database?",
        "What are the min and max temperatures recorded?", 
        "Show me the latitude, longitude, and temperature for the 5 deepest measurements",
        "What is the average surface temperature for each float where depth is less than 10 meters?",
        "Show me the geographical bounding box of all measurements"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*20} Test Question {i} {'='*20}")
        
        # Execute query and display results (enable debug for first question)
        debug_mode = (i == 1)
        result = executor.query_and_execute(question, max_rows=20)
        executor.display_results(result)
        
        # Add a small pause between queries
        import time
        time.sleep(1)
    
    print("\n🎉 All tests completed!")
    
    # Example of how to use the functions programmatically
    print("\n" + "="*50)
    print("📚 Example of programmatic usage:")
    print("="*50)
    
    # Method 1: Using the class directly
    result1 = executor.query_and_execute("How many profiles does each float have?", max_rows=10)
    if result1['success']:
        print(f"✅ Found {result1['total_rows']} results")
        print(f"First result: {result1['data'][0] if result1['data'] else 'No data'}")
    
    # Method 2: Using convenience functions
    result2 = query_database("What is the deepest measurement in the database?")
    print(f"Query result success: {result2['success']}")