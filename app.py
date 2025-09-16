import streamlit as st
import pandas as pd
from reg import RAGSQLQueryExecutor  # Import the main class from your script

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Argo Ocean Data Explorer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Main App UI ---
st.title("🌊 Argo Ocean Data Explorer")
st.markdown("""
Welcome to the Argo Ocean Data Explorer! Ask a question in plain English about the oceanographic data, 
and the AI will generate and execute the necessary SQL query to find the answer.
""")

# --- Caching the Executor ---
# This prevents re-initializing the model every time you interact with the app
@st.cache_resource
def get_executor():
    """Creates and caches the RAGSQLQueryExecutor instance."""
    try:
        executor = RAGSQLQueryExecutor()
        return executor
    except Exception as e:
        st.error(f"Failed to initialize the backend: {e}")
        st.info("Please ensure Ollama is running (`ollama serve`) and the model is downloaded (`ollama pull llama3.2`).")
        return None

executor = get_executor()

if executor:
    # --- User Input ---
    st.sidebar.header("Ask a Question")
    user_question = st.sidebar.text_area(
        "Enter your question here:",
        "What are the average temperature and salinity at depths greater than 1000 meters?",
        height=100
    )

    # Pre-defined example questions
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Example Questions:")
    example_questions = [
        "How many unique floats are in the database?",
        "What are the temperature patterns in the Southern Ocean?",
        "Show me the deepest measurements and their characteristics",
        "What is the salinity distribution at different depths?",
    ]
    for q in example_questions:
        if st.sidebar.button(q):
            user_question = q


    if st.sidebar.button("Get Answer"):
        if user_question:
            with st.spinner("Analyzing your question and querying the database..."):
                try:
                    # --- RAG Pipeline Execution ---
                    result = executor.query_with_rag(user_question, max_rows=50)

                    # --- Display Results ---
                    st.header("Results")

                    # Display AI Analysis First
                    st.subheader("🤖 AI Analysis")
                    if result.get("enhanced_response"):
                        st.markdown(result["enhanced_response"])
                    else:
                        st.warning("Could not generate an AI analysis.")

                    # Use columns for better layout
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Generated SQL Query")
                        st.code(result.get('query', 'N/A'), language='sql')

                    with col2:
                        st.subheader("Query Status")
                        if result['success']:
                            st.success(f"Query executed successfully. Found {result['total_rows']} rows.")
                        else:
                            st.error(f"Query failed: {result.get('error', 'Unknown error')}")


                    # Display Raw Data if successful
                    if result['success'] and result['data']:
                        st.subheader("📋 Raw Data Results")
                        df = pd.DataFrame(result['data'])
                        st.dataframe(df)
                        if result['truncated']:
                            st.info(f"Note: Results are truncated to the first {result['displayed_rows']} rows for display.")

                    # Display historical context
                    if result.get('context_insights'):
                        with st.expander("🧠 View Relevant Historical Insights"):
                            for insight_data in result['context_insights']:
                                st.markdown(f"**Insight (Similarity: {insight_data['similarity']:.2f}):**")
                                st.info(f"_{insight_data['insight']}_")
                                st.caption(f"From a previous question: \"{insight_data['metadata']['question']}\"")
                                st.markdown("---")


                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
        else:
            st.sidebar.warning("Please enter a question.")
else:
    st.error("The application backend could not be started. Please check the console for errors.")