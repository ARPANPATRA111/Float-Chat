# --- app.py ---
import streamlit as st
import pandas as pd
import plotly.express as px
from rag_system import RAGSQLQueryExecutor
from data_processing import ArgoDataProcessor

# Page configuration
st.set_page_config(page_title="ARGO Ocean Data Explorer", page_icon="🌊", layout="wide")

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .st-emotion-cache-1y4p8pa {
        padding-top: 2rem;
    }
    .st-emotion-cache-1v0mbdj {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# Initialize components using Streamlit's caching
@st.cache_resource
def get_executor():
    try:
        return RAGSQLQueryExecutor()
    except Exception as e:
        st.error(f"Failed to initialize AI Engine (Ollama): {e}", icon="🚨")
        st.info("Please ensure Ollama is running and the specified model is installed (`ollama pull llama3`)")
        return None

@st.cache_resource
def get_data_processor():
    return ArgoDataProcessor()

def create_visualizations(df: pd.DataFrame):
    if df.empty:
        st.warning("No data available for visualization.")
        return

    st.markdown("---")
    st.subheader("Interactive Visualizations")
    
    # Use tabs for different plot types
    map_tab, profile_tab, series_tab = st.tabs(["🗺️ Geospatial Map", "📈 Depth Profile", "📉 Time Series"])
    
    with map_tab:
        if 'lat' in df.columns and 'lon' in df.columns:
            color_opts = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col not in ['lat', 'lon']]
            color_by = st.selectbox("Color map points by:", options=color_opts, index=0 if 'temperature' in color_opts else 0)
            fig = px.scatter_mapbox(df, lat='lat', lon='lon', color=color_by,
                                    mapbox_style="open-street-map", zoom=2,
                                    hover_name='float_id', hover_data=df.columns,
                                    title=f"Float Locations colored by {color_by}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Geospatial data (lat, lon) not found in results.")

    with profile_tab:
        if 'depth' in df.columns:
            param_opts = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col != 'depth']
            param = st.selectbox("Select parameter for profile:", options=param_opts, index=0 if 'temperature' in param_opts else 0)
            fig = px.line(df, x=param, y='depth', color='float_id', title=f"{param.capitalize()} vs. Depth Profile")
            fig.update_yaxes(autorange="reversed") # Depth increases downwards
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Depth data not found in results.")
            
    with series_tab:
        if 'time' in df.columns:
            param_opts_ts = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
            param_ts = st.selectbox("Select parameter for time series:", options=param_opts_ts, index=0 if 'temperature' in param_opts_ts else 0)
            fig = px.line(df, x='time', y=param_ts, color='float_id', title=f"{param_ts.capitalize()} over Time")
            st.plotly_chart(fig, use_container_width=True)
        else:
             st.info("Time data not found in results.")


# Main app interface
st.title("🌊 Conversational ARGO Ocean Data Explorer")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for controls and examples
with st.sidebar:
    st.header("Controls")
    if st.button("Process New Data Files", type="primary"):
        with st.spinner("Processing ARGO NetCDF files... This may take a moment."):
            processor = get_data_processor()
            processor.process_directory()
        st.success("Data processing complete!")
        st.rerun()

    st.markdown("---")
    st.header("Example Questions")
    example_questions = [
        "Show me the latest temperature and salinity profiles.",
        "What are the average temperatures below 500m depth?",
        "List all floats that measure chlorophyll (chla).",
        "Compare salinity in the Arabian Sea vs the Bay of Bengal.",
        "What are the nearest floats to Goa, India (15.29° N, 73.91° E)?"
    ]
    for q in example_questions:
        if st.button(q):
            st.session_state.user_input = q
            st.rerun()

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "data" in message:
            df = pd.DataFrame(message["data"])
            with st.expander("View Data & Download"):
                st.dataframe(df)
                # NEW: Added download button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                   "Download as CSV",
                   csv,
                   f"argo_query_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                   "text/csv",
                   key=f'download-csv-{message["timestamp"]}' # Unique key
                )
            create_visualizations(df)
        if "query" in message:
            with st.expander("Generated SQL Query"):
                st.code(message["query"], language="sql")

# Main chat input
if prompt := st.chat_input("Ask about ARGO ocean data...", key="user_input"):
    executor = get_executor()
    if executor:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = executor.query_with_rag(prompt)
                
                message = {
                    "role": "assistant",
                    "content": response.get("enhanced_response", "An error occurred."),
                    "timestamp": pd.Timestamp.now()
                }

                if response["success"] and response.get("data"):
                    message["data"] = response["data"]
                    message["query"] = response["generated_query"]
                
                st.session_state.messages.append(message)
                st.rerun()