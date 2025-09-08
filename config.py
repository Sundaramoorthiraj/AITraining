"""
Configuration settings for the Local RAG System
"""
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Config:
    """Configuration class for RAG system parameters"""
    
    # Model Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "microsoft/DialoGPT-medium"  # Fallback for CPU
    quantized_llm_model: str = "TheBloke/Llama-2-7B-Chat-GGML"  # For quantized inference
    
    # Document Processing
    chunk_size: int = 512
    chunk_overlap: int = 50
    supported_formats: List[str] = None
    
    # Vector Store
    vector_store_path: str = "./vector_store"
    similarity_threshold: float = 0.7
    top_k_results: int = 5
    
    # LLM Inference
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    use_cpu: bool = True
    
    # Paths
    documents_path: str = "./documents"
    cache_dir: str = "./cache"
    
    def __post_init__(self):
        """Initialize default values after dataclass creation"""
        if self.supported_formats is None:
            self.supported_formats = ['.pdf', '.docx', '.doc', '.xlsx', '.xls', '.txt']
        
        # Create directories if they don't exist
        os.makedirs(self.vector_store_path, exist_ok=True)
        os.makedirs(self.documents_path, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create config from environment variables"""
        return cls(
            embedding_model=os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
            llm_model=os.getenv('LLM_MODEL', 'microsoft/DialoGPT-medium'),
            chunk_size=int(os.getenv('CHUNK_SIZE', '512')),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '50')),
            vector_store_path=os.getenv('VECTOR_STORE_PATH', './vector_store'),
            documents_path=os.getenv('DOCUMENTS_PATH', './documents'),
            use_cpu=os.getenv('USE_CPU', 'true').lower() == 'true'
        )


# Global configuration instance
config = Config()