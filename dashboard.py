import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# Ensure these imports are correct for your project structure
from rag_system import RAGSQLQueryExecutor
from data_processing import ArgoDataProcessor

# Page configuration
st.set_page_config(
    page_title="FloatChat - ARGO Ocean Data Explorer", 
    page_icon="🌊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional appearance
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        padding-top: 1rem;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #0066cc 0%, #004d99 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 102, 204, 0.2);
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    .sidebar-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    
    .sidebar-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        border-bottom: 2px solid #0066cc;
        padding-bottom: 0.5rem;
    }
    
    /* Stats Cards */
    .stats-container {
        display: flex;
        gap: 1rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    
    .stat-card {
        flex: 1;
        min-width: 200px;
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
        border-left: 4px solid #0066cc;
        transition: transform 0.2s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(0,0,0,0.1);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0066cc;
        margin: 0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #64748b;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Chat Interface */
    .chat-container {
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .chat-header {
        background: linear-gradient(90deg, #0066cc 0%, #0052a3 100%);
        color: white;
        padding: 1rem 1.5rem;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Visualization Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        border-radius: 10px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 20px;
        background-color: transparent;
        border-radius: 8px;
        color: #64748b;
        font-weight: 500;
        border: none;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #0066cc;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Example Questions */
    .example-question {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
        cursor: pointer;
        font-size: 0.9rem;
    }
    
    .example-question:hover {
        border-color: #0066cc;
        background: #f8fafc;
        transform: translateX(4px);
    }
    
    /* Status Indicators */
    .status-success {
        color: #059669;
        background: #d1fae5;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #059669;
    }
    
    .status-warning {
        color: #d97706;
        background: #fef3c7;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #d97706;
    }
    
    .status-error {
        color: #dc2626;
        background: #fee2e2;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc2626;
    }
    
    /* Fallback Tips */
    .fallback-tips {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #f59e0b;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .fallback-tips h4 {
        color: #92400e;
        margin-top: 0;
        font-weight: 600;
    }
    
    .fallback-tips ul {
        color: #78350f;
        margin-bottom: 0;
    }
    
    /* Download Button Styling */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #059669 0%, #047857 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.3);
    }
    
    /* Data Table Styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Code Block Styling */
    .stCodeBlock {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    /* Loading Spinner */
    .stSpinner {
        color: #0066cc;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components using Streamlit's caching
@st.cache_resource
def get_executor():
    try:
        return RAGSQLQueryExecutor()
    except Exception as e:
        st.error(f"Failed to initialize AI Engine (Ollama): {e}")
        st.info("Please ensure Ollama is running and the specified model is installed.")
        return None

@st.cache_resource
def get_data_processor():
    return ArgoDataProcessor()

# Enhanced visualization function
def create_visualizations(df: pd.DataFrame, unique_id: str):
    if df.empty:
        st.warning("No data available for visualization.")
        return

    st.markdown("---")
    
    # Visualization header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📊 Interactive Visualizations")
    with col2:
        st.metric("Data Points", f"{len(df):,}")
    
    # Enhanced tabs with better styling
    map_tab, profile_tab, series_tab, stats_tab = st.tabs([
        "🗺️ Geospatial Map", 
        "📈 Depth Profile", 
        "📉 Time Series",
        "📊 Statistics"
    ])
    
    with map_tab:
        if 'lat' in df.columns and 'lon' in df.columns:
            map_df = df.drop_duplicates(subset=['lat', 'lon'])
            color_opts = [col for col in map_df.columns if map_df[col].dtype in ['float64', 'int64', 'float32'] and col not in ['lat', 'lon']]
            
            if not color_opts:
                color_opts = [map_df.columns[0]]

            col1, col2 = st.columns([3, 1])
            with col2:
                color_by = st.selectbox(
                    "Color points by:", 
                    options=color_opts, 
                    index=0, 
                    key=f"map_color_{unique_id}"
                )
            
            fig = px.scatter_mapbox(
                map_df, 
                lat='lat', 
                lon='lon', 
                color=color_by,
                mapbox_style="carto-positron",
                zoom=2,
                hover_name='float_id' if 'float_id' in map_df.columns else None, 
                hover_data={col: True for col in map_df.columns[:5]},
                title=f"Ocean Float Locations ({len(map_df)} points)",
                color_continuous_scale="Viridis",
                height=600
            )
            fig.update_layout(
                font=dict(family="Inter, sans-serif"),
                title_font_size=16,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Map statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Unique Locations", f"{len(map_df)}")
            with col2:
                if 'float_id' in map_df.columns:
                    st.metric("Unique Floats", f"{map_df['float_id'].nunique()}")
            with col3:
                lat_range = map_df['lat'].max() - map_df['lat'].min()
                st.metric("Latitude Range", f"{lat_range:.2f}°")
        else:
            st.info("🗺️ Geospatial data (latitude/longitude) not found in results.")

    with profile_tab:
        if 'depth' in df.columns:
            param_opts = [col for col in df.columns if df[col].dtype in ['float64', 'int64', 'float32'] and col != 'depth']
            
            if not param_opts:
                st.info("No numeric parameters found for depth profile.")
            else:
                col1, col2 = st.columns([3, 1])
                with col2:
                    param = st.selectbox(
                        "Parameter:", 
                        options=param_opts, 
                        index=0,
                        key=f"profile_param_{unique_id}"
                    )
                
                # Create unique profile identifier
                if 'float_id' in df.columns and 'profile_number' in df.columns:
                    df['profile_id'] = df['float_id'].astype(str) + '_' + df['profile_number'].astype(str)
                    plot_df = df.sort_values(by='depth')
                    color_col = 'profile_id'
                else:
                    plot_df = df.sort_values(by='depth')
                    color_col = 'float_id' if 'float_id' in df.columns else None
                
                fig = px.line(
                    plot_df, 
                    x=param, 
                    y='depth', 
                    color=color_col, 
                    title=f"{param.capitalize()} vs. Depth Profile",
                    height=600,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_yaxes(autorange="reversed", title="Depth (m)")
                fig.update_xaxes(title=f"{param.capitalize()}")
                fig.update_layout(
                    font=dict(family="Inter, sans-serif"),
                    title_font_size=16,
                    hovermode='closest'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Profile statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Max Depth", f"{df['depth'].max():.1f} m")
                with col2:
                    st.metric("Depth Range", f"{df['depth'].max() - df['depth'].min():.1f} m")
                with col3:
                    st.metric("Avg " + param.title(), f"{df[param].mean():.2f}")
        else:
            st.info("📈 Depth data not found in results.")
            
    with series_tab:
        if 'time' in df.columns:
            param_opts_ts = [col for col in df.columns if df[col].dtype in ['float64', 'int64', 'float32']]
            
            if not param_opts_ts:
                st.info("No numeric parameters found for time series.")
            else:
                col1, col2 = st.columns([3, 1])
                with col2:
                    param_ts = st.selectbox(
                        "Parameter:", 
                        options=param_opts_ts, 
                        index=0,
                        key=f"series_param_{unique_id}"
                    )
                
                color_col = 'float_id' if 'float_id' in df.columns else None
                
                fig = px.line(
                    df, 
                    x='time', 
                    y=param_ts, 
                    color=color_col, 
                    title=f"{param_ts.capitalize()} Time Series",
                    height=600,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(
                    font=dict(family="Inter, sans-serif"),
                    title_font_size=16,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Time series statistics
                if pd.api.types.is_datetime64_any_dtype(df['time']):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        time_range = df['time'].max() - df['time'].min()
                        st.metric("Time Span", f"{time_range.days} days")
                    with col2:
                        st.metric("Data Points", f"{len(df)}")
                    with col3:
                        st.metric("Trend", "📈 Positive" if df[param_ts].iloc[-1] > df[param_ts].iloc[0] else "📉 Negative")
        else:
             st.info("📉 Time data not found in results.")
    
    with stats_tab:
        st.subheader("📊 Data Statistics")
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_df = df[numeric_cols].describe().round(3)
            st.dataframe(stats_df, use_container_width=True)
            
            # Correlation matrix for numeric columns
            if len(numeric_cols) > 1:
                st.subheader("🔗 Correlation Matrix")
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="Parameter Correlations"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns found for statistical analysis.")

# Main application
def main():
    # Header section
    st.markdown("""
    <div class="main-header">
        <h1>🌊 FloatChat</h1>
        <p>Advanced ARGO Ocean Data Explorer with AI-Powered Insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-header">🚀 System Controls</div>
        """, unsafe_allow_html=True)
        
        # System status
        executor = get_executor()
        if executor:
            st.markdown('<div class="status-success">✅ AI System Online</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">❌ AI System Offline</div>', unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Data processing section
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-header">📊 Data Management</div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Process New Data Files", type="primary", use_container_width=True):
            with st.spinner("Processing ARGO NetCDF files... This may take a moment."):
                processor = get_data_processor()
                try:
                    processor.process_directory()
                    st.success("✅ Data processing complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Example queries section
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-header">💡 Example Queries</div>
        """, unsafe_allow_html=True)
        
        example_questions = [
            "Show me the latest temperature and salinity profiles",
            "What are the average temperatures below 500m depth?",
            "Find the nearest floats to Mumbai, India",
            "Plot temperature vs depth for recent data",
            "Show salinity trends over the past month",
            "What's the temperature range at 1000m depth?"
        ]
        
        for i, q in enumerate(example_questions):
            if st.button(q, key=f"example_{i}", use_container_width=True, help="Click to use this example query"):
                st.session_state.user_input = q
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Quick stats section (if we have data)
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-header">📈 Quick Stats</div>
        """, unsafe_allow_html=True)
        
        try:
            if executor:
                # Try to get some quick stats
                stats_result = executor.db_manager.execute_query("SELECT COUNT(*) as total_records, COUNT(DISTINCT float_id) as unique_floats FROM argo_profiles LIMIT 1")
                if stats_result["success"] and stats_result["data"]:
                    stats = stats_result["data"][0]
                    st.metric("Total Records", f"{stats['total_records']:,}")
                    st.metric("Active Floats", f"{stats['unique_floats']:,}")
        except:
            st.info("Connect to database to view stats")
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Main chat interface
    st.markdown("""
    <div class="chat-container">
        <div class="chat-header">
            💬 Conversational Ocean Data Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Handle data visualization for valid queries
            if "data" in message and message["data"] and not message.get("fallback", False):
                df = pd.DataFrame(message["data"])
                unique_key = str(message["timestamp"]).replace(" ", "_").replace(":", "-")

                # Data summary cards
                if len(df) > 0:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Records", f"{len(df):,}")
                    with col2:
                        if 'float_id' in df.columns:
                            st.metric("Floats", f"{df['float_id'].nunique()}")
                    with col3:
                        if 'depth' in df.columns:
                            st.metric("Max Depth", f"{df['depth'].max():.0f}m")
                    with col4:
                        if 'time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['time']):
                            st.metric("Latest", df['time'].max().strftime("%Y-%m-%d"))

                # Data table and download
                with st.expander("📋 View Data & Download", expanded=False):
                    st.dataframe(df, use_container_width=True, height=300)
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                           "📥 Download CSV",
                           csv,
                           f"argo_query_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           "text/csv",
                           key=f'download_{unique_key}'
                        )
                    with col2:
                        json_data = df.to_json(orient='records', date_format='iso').encode('utf-8')
                        st.download_button(
                           "📥 Download JSON",
                           json_data,
                           f"argo_query_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                           "application/json",
                           key=f'download_json_{unique_key}'
                        )
                
                # Create visualizations
                create_visualizations(df, unique_key)

            # Show SQL query for valid queries
            if "query" in message and not message.get("fallback", False):
                with st.expander("🔍 Generated SQL Query", expanded=False):
                    st.code(message["query"], language="sql")
            
            # Show helpful tips for fallback responses
            elif message.get("fallback", False):
                st.markdown("""
                <div class="fallback-tips">
                    <h4>💡 Try asking specific questions about ocean data!</h4>
                    <p><strong>Example queries you can try:</strong></p>
                    <ul>
                        <li>Show me temperature profiles for the latest data</li>
                        <li>What are the nearest floats to Mumbai, India?</li>
                        <li>Plot salinity vs depth for recent measurements</li>
                        <li>Find temperature data below 1000 meters depth</li>
                        <li>Show me floats in the Arabian Sea</li>
                        <li>What's the average salinity at 500m depth?</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask me about ARGO ocean data... 🌊", key="user_input"):
        executor = get_executor()
        if executor:
            # Add user message
            st.session_state.messages.append({
                "role": "user", 
                "content": prompt, 
                "timestamp": pd.Timestamp.now()
            })
            
            with st.chat_message("user"):
                st.markdown(prompt)

            # Process and display response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Analyzing your query..."):
                    response = executor.query_with_rag(prompt)
                    
                    message = {
                        "role": "assistant",
                        "content": response.get("enhanced_response", "An error occurred while processing your request."),
                        "timestamp": pd.Timestamp.now()
                    }

                    # Add data and query for valid ocean data queries
                    if response["success"] and response.get("data") and not response.get("fallback_used", False):
                        message["data"] = response["data"]
                        message["query"] = response["generated_query"]
                    elif response.get("fallback_used", False):
                        message["fallback"] = True
                    
                    st.session_state.messages.append(message)
                    st.rerun()
        else:
            st.error("❌ AI system is not available. Please check your Ollama installation.")

    # Footer
    st.markdown("""
    ---
    <div style='text-align: center; color: #64748b; font-size: 0.9rem; padding: 2rem 0;'>
        <p>FloatChat - Advanced Ocean Data Analysis Platform | Powered by ARGO Global Ocean Observing System</p>
        <p>Built with Streamlit • PostgreSQL • Ollama • ChromaDB</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()