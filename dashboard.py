# --- dashboard.py ---
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Ensure these imports are correct for your project structure
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
        st.info("Please ensure Ollama is running and the specified model is installed.")
        return None

@st.cache_resource
def get_data_processor():
    return ArgoDataProcessor()

# --- MODIFIED: The Visualization Function ---
def create_visualizations(df: pd.DataFrame, unique_id: str):
    if df.empty:
        st.warning("No data available for visualization.")
        return

    st.markdown("---")
    st.subheader("Interactive Visualizations")
    
    map_tab, profile_tab, series_tab = st.tabs(["🗺️ Geospatial Map", "📈 Depth Profile", "📉 Time Series"])
    
    with map_tab:
        if 'lat' in df.columns and 'lon' in df.columns:
            # Drop duplicate locations for a cleaner map
            map_df = df.drop_duplicates(subset=['lat', 'lon'])
            color_opts = [col for col in map_df.columns if map_df[col].dtype in ['float64', 'int64', 'float32'] and col not in ['lat', 'lon']]
            if not color_opts:
                color_opts = [map_df.columns[0]] 

            color_by = st.selectbox(
                "Color map points by:", 
                options=color_opts, 
                index=0, 
                key=f"map_color_{unique_id}"
            )
            
            fig = px.scatter_mapbox(map_df, lat='lat', lon='lon', color=color_by,
                                    mapbox_style="open-street-map", zoom=2,
                                    hover_name='float_id' if 'float_id' in map_df.columns else None, 
                                    hover_data=map_df.columns,
                                    title=f"Float Locations colored by {color_by}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Geospatial data (lat, lon) not found in results for this query.")

    with profile_tab:
        if 'depth' in df.columns:
            param_opts = [col for col in df.columns if df[col].dtype in ['float64', 'int64', 'float32'] and col != 'depth']
            if not param_opts:
                st.info("No numeric parameters found for depth profile.")
            else:
                param = st.selectbox(
                    "Select parameter for profile:", 
                    options=param_opts, 
                    index=0,
                    key=f"profile_param_{unique_id}"
                )
                
                # --- NEW: Create a unique ID for each profile and sort the data ---
                if 'float_id' in df.columns and 'profile_number' in df.columns:
                    # Create a unique identifier for each individual dive
                    df['profile_id'] = df['float_id'].astype(str) + '_' + df['profile_number'].astype(str)
                    # Sort by depth to ensure lines are drawn correctly
                    plot_df = df.sort_values(by='depth')
                    color_col = 'profile_id'
                else:
                    plot_df = df.sort_values(by='depth')
                    color_col = 'float_id' if 'float_id' in df.columns else None
                
                fig = px.line(plot_df, x=param, y='depth', color=color_col, title=f"{param.capitalize()} vs. Depth Profile")
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Depth data not found in results for this query.")
            
    with series_tab:
        # (No changes needed here, but kept for completeness)
        if 'time' in df.columns:
            param_opts_ts = [col for col in df.columns if df[col].dtype in ['float64', 'int64', 'float32']]
            if not param_opts_ts:
                st.info("No numeric parameters found for time series.")
            else:
                param_ts = st.selectbox(
                    "Select parameter for time series:", 
                    options=param_opts_ts, 
                    index=0,
                    key=f"series_param_{unique_id}"
                )
                
                color_col = 'float_id' if 'float_id' in df.columns else None

                fig = px.line(df, x='time', y=param_ts, color=color_col, title=f"{param_ts.capitalize()} over Time")
                st.plotly_chart(fig, use_container_width=True)
        else:
             st.info("Time data not found in results for this query.")

# --- Main app loop (no changes needed from the previous version) ---
st.title("🌊 Conversational ARGO Ocean Data Explorer")

if "messages" not in st.session_state:
    st.session_state.messages = []

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
        "What are the nearest floats to Goa, India (15.29° N, 73.91° E)?",
        "Plot a depth profile of temperature for all floats between January 1st and 5th, 2024."
    ]
    for q in example_questions:
        if st.button(q, key=q):
            st.session_state.user_input = q
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "data" in message and message["data"]:
            df = pd.DataFrame(message["data"])
            unique_key = str(message["timestamp"])

            with st.expander("View Data & Download"):
                st.dataframe(df)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                   "Download as CSV",
                   csv,
                   f"argo_query_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                   "text/csv",
                   key=f'download_{unique_key}'
                )
            
            create_visualizations(df, unique_key)

        if "query" in message:
            with st.expander("Generated SQL Query"):
                st.code(message["query"], language="sql")

if prompt := st.chat_input("Ask about ARGO ocean data...", key="user_input"):
    executor = get_executor()
    if executor:
        st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": pd.Timestamp.now()})
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