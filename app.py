"""
Streamlit web interface for the Local RAG System
"""
import streamlit as st
import logging
from pathlib import Path
import tempfile
import os

from src.rag_pipeline import RAGPipeline
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Local RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_pipeline():
    """Initialize the RAG pipeline with caching"""
    try:
        config = Config()
        rag = RAGPipeline(config)
        return rag, config
    except Exception as e:
        st.error(f"Error initializing RAG pipeline: {e}")
        return None, None

def display_system_stats(rag):
    """Display system statistics"""
    if rag is None:
        return
    
    stats = rag.get_system_stats()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("System Status", "✅ Ready" if stats['initialized'] else "❌ Not Ready")
    
    with col2:
        if stats['vector_store']['status'] != 'empty':
            st.metric("Documents", stats['vector_store']['total_chunks'])
        else:
            st.metric("Documents", "0")
    
    with col3:
        if stats['vector_store']['status'] != 'empty':
            st.metric("Embeddings", stats['vector_store']['total_embeddings'])
        else:
            st.metric("Embeddings", "0")

def upload_documents():
    """Handle document upload"""
    st.subheader("📁 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['pdf', 'docx', 'doc', 'xlsx', 'xls', 'txt'],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, DOC, XLSX, XLS, TXT"
    )
    
    if uploaded_files:
        # Create temporary directory for uploaded files
        temp_dir = tempfile.mkdtemp()
        
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        return temp_dir, uploaded_files
    
    return None, None

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">📚 Local RAG System</div>', unsafe_allow_html=True)
    st.markdown("**Document Question & Answering with Local LLMs**")
    
    # Initialize RAG pipeline
    rag, config = initialize_rag_pipeline()
    
    if rag is None:
        st.error("Failed to initialize the RAG system. Please check the logs.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Control")
        
        # System statistics
        st.subheader("System Status")
        display_system_stats(rag)
        
        # Document management
        st.subheader("📄 Document Management")
        
        if st.button("🔄 Refresh System Stats"):
            st.rerun()
        
        if st.button("🗑️ Clear Vector Store"):
            if st.session_state.get('confirm_clear', False):
                rag.clear_vector_store()
                st.success("Vector store cleared!")
                st.session_state.confirm_clear = False
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("Click again to confirm clearing the vector store")
        
        # Configuration
        st.subheader("⚙️ Configuration")
        st.text(f"Embedding Model: {config.embedding_model}")
        st.text(f"LLM Model: {config.llm_model}")
        st.text(f"Chunk Size: {config.chunk_size}")
        st.text(f"Top K Results: {config.top_k_results}")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "📁 Manage Documents", "📊 System Info"])
    
    with tab1:
        st.markdown('<div class="sub-header">💬 Ask Questions</div>', unsafe_allow_html=True)
        
        # Check if documents are processed
        stats = rag.get_system_stats()
        if not stats['documents_processed']:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ No documents processed yet!</strong><br>
                Please upload and process documents in the "Manage Documents" tab before asking questions.
            </div>
            """, unsafe_allow_html=True)
        else:
            # Question input
            question = st.text_area(
                "Enter your question:",
                placeholder="What would you like to know about your documents?",
                height=100
            )
            
            # Advanced options
            with st.expander("🔧 Advanced Options"):
                col1, col2 = st.columns(2)
                with col1:
                    top_k = st.slider("Number of relevant chunks", 1, 10, config.top_k_results)
                with col2:
                    show_sources = st.checkbox("Show sources", value=True)
                    show_context = st.checkbox("Show context chunks", value=False)
            
            # Process question
            if st.button("🔍 Ask Question", type="primary") and question:
                with st.spinner("Processing your question..."):
                    try:
                        result = rag.query(question, top_k=top_k)
                        
                        # Display answer
                        st.markdown("### 💡 Answer")
                        st.markdown(result['answer'])
                        
                        # Display sources
                        if show_sources and result['sources']:
                            st.markdown("### 📚 Sources")
                            for source in result['sources']:
                                st.text(f"• {Path(source).name}")
                        
                        # Display context chunks
                        if show_context and result['context_chunks']:
                            st.markdown("### 🔍 Context Chunks")
                            for i, chunk in enumerate(result['context_chunks'], 1):
                                with st.expander(f"Chunk {i} (Score: {result['similarity_scores'][i-1]:.3f})"):
                                    st.text(f"Source: {Path(chunk['source']).name}")
                                    st.text(chunk['text'])
                    
                    except Exception as e:
                        st.error(f"Error processing question: {e}")
    
    with tab2:
        st.markdown('<div class="sub-header">📁 Manage Documents</div>', unsafe_allow_html=True)
        
        # Document upload section
        st.markdown("### Upload New Documents")
        temp_dir, uploaded_files = upload_documents()
        
        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} files:")
            for file in uploaded_files:
                st.text(f"• {file.name}")
            
            if st.button("🚀 Process Uploaded Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    try:
                        rag.ingest_documents(temp_dir)
                        st.success("Documents processed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing documents: {e}")
        
        # Local documents section
        st.markdown("### Process Local Documents")
        local_path = st.text_input(
            "Path to documents directory:",
            value=config.documents_path,
            help="Enter the path to a directory containing your documents"
        )
        
        if st.button("📂 Process Local Documents"):
            if os.path.exists(local_path):
                with st.spinner("Processing local documents..."):
                    try:
                        rag.ingest_documents(local_path)
                        st.success("Local documents processed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing local documents: {e}")
            else:
                st.error(f"Directory does not exist: {local_path}")
        
        # Document statistics
        if stats['documents_processed']:
            st.markdown("### 📊 Document Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Chunks", stats['vector_store']['total_chunks'])
            with col2:
                st.metric("Total Embeddings", stats['vector_store']['total_embeddings'])
            with col3:
                st.metric("Embedding Dimension", stats['vector_store']['embedding_dimension'])
    
    with tab3:
        st.markdown('<div class="sub-header">📊 System Information</div>', unsafe_allow_html=True)
        
        # System statistics
        stats = rag.get_system_stats()
        
        st.markdown("### 🔧 System Status")
        col1, col2 = st.columns(2)
        
        with col1:
            st.json({
                "Initialized": stats['initialized'],
                "Documents Processed": stats['documents_processed'],
                "Vector Store Status": stats['vector_store']['status']
            })
        
        with col2:
            if stats['vector_store']['status'] != 'empty':
                st.json({
                    "Total Embeddings": stats['vector_store']['total_embeddings'],
                    "Embedding Dimension": stats['vector_store']['embedding_dimension'],
                    "Total Chunks": stats['vector_store']['total_chunks']
                })
        
        # Model information
        st.markdown("### 🤖 Model Information")
        
        if 'embedding_model' in stats:
            st.markdown("**Embedding Model:**")
            st.text(f"Name: {stats['embedding_model']['name']}")
            st.text(f"Dimension: {stats['embedding_model']['dimension']}")
        
        if stats['llm_model']['status'] == 'loaded':
            st.markdown("**LLM Model:**")
            st.text(f"Name: {stats['llm_model']['model_name']}")
            st.text(f"Parameters: {stats['llm_model']['num_parameters']:,}")
            st.text(f"Size: {stats['llm_model']['model_size_mb']:.1f} MB")
            st.text(f"Device: {stats['llm_model']['device']}")
        
        # Configuration
        st.markdown("### ⚙️ Configuration")
        config_dict = {
            "Chunk Size": config.chunk_size,
            "Chunk Overlap": config.chunk_overlap,
            "Top K Results": config.top_k_results,
            "Similarity Threshold": config.similarity_threshold,
            "Max Tokens": config.max_tokens,
            "Temperature": config.temperature,
            "Use CPU": config.use_cpu
        }
        st.json(config_dict)

if __name__ == "__main__":
    main()