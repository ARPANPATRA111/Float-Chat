# app.py

import streamlit as st
import pandas as pd
from backend import query_database, SQLQueryExecutor  # Import our backend functions

# --- Page Configuration ---
st.set_page_config(
    page_title="FloatChat 🌊",
    page_icon="🤖",
    layout="wide"
)

# --- Initialize session state ---
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# --- UI Elements ---
st.title("FloatChat 🌊🤖")
st.caption("Your AI-Powered Conversational Interface for ARGO Ocean Data")

# Create two columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Ask Your Question")
    
    # Get user input
    user_question = st.text_input(
        "Ask a question about the ARGO float data:",
        placeholder="e.g., How many floats are in the database? What's the deepest measurement?"
    )

    # --- Backend Logic ---
    if st.button("Get Answer", type="primary", use_container_width=True):
        if user_question:
            with st.spinner("🤖 Analyzing your question and querying the database..."):
                try:
                    # Call the backend function
                    result = query_database(user_question, max_rows=100)
                    
                    # Add to history
                    st.session_state.query_history.append({
                        "question": user_question,
                        "result": result
                    })
                    
                    if result['success']:
                        st.success("✅ Query executed successfully!")
                        
                        # Display the generated SQL query
                        with st.expander("🔍 Generated SQL Query", expanded=False):
                            st.code(result['query'], language='sql')
                        
                        # Display results
                        st.subheader("📊 Results")
                        
                        if result['data']:
                            # Create DataFrame for display
                            df = pd.DataFrame(result['data'])
                            
                            # Show metadata
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Total Rows", result['total_rows'])
                            with col_b:
                                st.metric("Displayed Rows", result['displayed_rows'])
                            with col_c:
                                if result['truncated']:
                                    st.warning(f"⚠️ Truncated")
                                else:
                                    st.success("✅ Complete")
                            
                            # Display the data table
                            st.dataframe(df, use_container_width=True)
                            
                            # Download button for results
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name=f"argo_query_results.csv",
                                mime="text/csv"
                            )
                            
                        else:
                            st.info("📋 No data returned from the query.")
                            
                    else:
                        st.error(f"❌ Query failed: {result['error']}")
                        
                        # Still show the attempted query for debugging
                        if result['query']:
                            with st.expander("🔍 Attempted SQL Query"):
                                st.code(result['query'], language='sql')
                        
                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")
        else:
            st.warning("⚠️ Please enter a question.")

with col2:
    st.subheader("💡 Example Questions")
    
    example_questions = [
        "How many unique floats are in the database?",
        "What are the min and max temperatures recorded?",
        "Show me the 10 deepest measurements",
        "What's the average surface temperature by float?",
        "Show me the geographical bounding box",
        "How many profiles does each float have?",
        "What's the temperature range at 1000m depth?",
        "Show me measurements from the Southern Ocean"
    ]
    
    for i, example in enumerate(example_questions):
        if st.button(example, key=f"example_{i}", use_container_width=True):
            st.session_state.temp_question = example
            # Auto-fill the text input
            user_question = example
            # Trigger the query automatically
            with st.spinner("🤖 Processing example question..."):
                try:
                    result = query_database(example, max_rows=100)
                    st.session_state.query_history.append({
                        "question": example,
                        "result": result
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error with example question: {e}")

# --- Query History Section ---
if st.session_state.query_history:
    st.markdown("---")
    st.subheader("📚 Query History")
    
    # Option to clear history
    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.query_history = []
        st.rerun()
    
    # Display history in reverse order (most recent first)
    for i, entry in enumerate(reversed(st.session_state.query_history)):
        with st.expander(f"Q{len(st.session_state.query_history)-i}: {entry['question'][:50]}{'...' if len(entry['question']) > 50 else ''}"):
            
            result = entry['result']
            
            # Show question and SQL
            st.write("**Question:**", entry['question'])
            st.code(result.get('query', 'No query generated'), language='sql')
            
            # Show results or error
            if result['success'] and result['data']:
                st.write(f"**Results:** {result['total_rows']} rows")
                df_history = pd.DataFrame(result['data'])
                st.dataframe(df_history.head(10), use_container_width=True)  # Show first 10 rows in history
                
                if len(result['data']) > 10:
                    st.caption(f"Showing first 10 of {len(result['data'])} rows")
            elif result['success']:
                st.info("Query executed successfully but returned no data.")
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")

# --- Sidebar with Database Info ---
with st.sidebar:
    st.header("🗄️ Database Info")
    
    # Show database statistics
    if st.button("📊 Show Database Stats"):
        with st.spinner("Loading database statistics..."):
            try:
                executor = SQLQueryExecutor()
                
                # Get basic stats
                stats_queries = {
                    "Total Records": "SELECT COUNT(*) as count FROM argo_profiles",
                    "Unique Floats": "SELECT COUNT(DISTINCT float_id) as count FROM argo_profiles",
                    "Date Range": "SELECT MIN(time) as min_date, MAX(time) as max_date FROM argo_profiles",
                    "Depth Range": "SELECT MIN(depth) as min_depth, MAX(depth) as max_depth FROM argo_profiles"
                }
                
                for stat_name, query in stats_queries.items():
                    result = executor.execute_query(query)
                    if result['success'] and result['data']:
                        st.metric(stat_name, result['data'][0])
                        
            except Exception as e:
                st.error(f"Could not load database stats: {e}")
    
    st.markdown("---")
    st.markdown("""
    ### 🔍 Tips for Better Queries:
    
    - Be specific about what data you want
    - Mention columns like temperature, depth, lat, lon
    - Use terms like "deepest", "surface", "average"
    - Ask about geographical regions
    - Inquire about specific floats or time periods
    
    ### 📋 Available Columns:
    - `float_id`: Float identifier
    - `profile_number`: Profile number
    - `time`: Measurement timestamp
    - `lat`, `lon`: Geographic coordinates  
    - `depth`: Measurement depth (meters)
    - `temperature`: Water temperature (°C)
    - `salinity`: Water salinity (PSU)
    """)
    
    st.markdown("---")
    st.caption("Built with Streamlit & Google Gemini 🤖")