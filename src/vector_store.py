"""
Vector store module using FAISS for efficient similarity search
"""
import os
import pickle
import logging
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from config import Config

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS-based vector store for efficient similarity search"""
    
    def __init__(self, config: Config):
        self.config = config
        self.index = None
        self.chunks = []
        self.embedding_dim = None
        self.index_path = os.path.join(config.vector_store_path, "faiss_index")
        self.metadata_path = os.path.join(config.vector_store_path, "metadata.pkl")
    
    def create_index(self, embedding_dim: int) -> None:
        """Create a new FAISS index"""
        self.embedding_dim = embedding_dim
        
        # Use IndexFlatIP for cosine similarity (inner product)
        # Normalize embeddings for cosine similarity
        self.index = faiss.IndexFlatIP(embedding_dim)
        
        logger.info(f"Created FAISS index with dimension {embedding_dim}")
    
    def add_embeddings(self, chunks: List[Dict[str, Any]]) -> None:
        """Add embeddings and metadata to the vector store"""
        if not chunks:
            logger.warning("No chunks provided to add to vector store")
            return
        
        # Extract embeddings
        embeddings = np.array([chunk['embedding'] for chunk in chunks])
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index if it doesn't exist
        if self.index is None:
            self.create_index(embeddings.shape[1])
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.chunks.extend(chunks)
        
        logger.info(f"Added {len(chunks)} embeddings to vector store. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = None) -> List[Dict[str, Any]]:
        """Search for similar embeddings"""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
        
        if top_k is None:
            top_k = self.config.top_k_results
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            result = {
                'chunk': self.chunks[idx],
                'score': float(score),
                'index': int(idx)
            }
            results.append(result)
        
        # Filter by similarity threshold
        threshold = self.config.similarity_threshold
        filtered_results = [r for r in results if r['score'] >= threshold]
        
        logger.info(f"Found {len(filtered_results)} results above threshold {threshold}")
        return filtered_results
    
    def save(self) -> None:
        """Save the vector store to disk"""
        if self.index is None:
            logger.warning("No index to save")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(self.config.vector_store_path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        
        # Save metadata
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        logger.info(f"Saved vector store to {self.config.vector_store_path}")
    
    def load(self) -> bool:
        """Load the vector store from disk"""
        try:
            # Check if files exist
            if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
                logger.info("No existing vector store found")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(self.index_path)
            
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                self.chunks = pickle.load(f)
            
            self.embedding_dim = self.index.d
            
            logger.info(f"Loaded vector store with {self.index.ntotal} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        if self.index is None:
            return {"status": "empty"}
        
        return {
            "status": "loaded",
            "total_embeddings": self.index.ntotal,
            "embedding_dimension": self.index.d,
            "index_type": type(self.index).__name__,
            "total_chunks": len(self.chunks)
        }
    
    def clear(self) -> None:
        """Clear the vector store"""
        self.index = None
        self.chunks = []
        self.embedding_dim = None
        
        # Remove files if they exist
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        
        logger.info("Cleared vector store")
    
    def rebuild_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Rebuild the entire index from scratch"""
        logger.info("Rebuilding vector store index")
        
        # Clear existing index
        self.clear()
        
        # Add all chunks
        self.add_embeddings(chunks)
        
        # Save the new index
        self.save()
        
        logger.info("Vector store rebuild complete")


class VectorStoreManager:
    """High-level manager for vector store operations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.vector_store = FAISSVectorStore(config)
    
    def ingest_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """Ingest document chunks into the vector store"""
        if not chunks:
            logger.warning("No chunks to ingest")
            return
        
        logger.info(f"Ingesting {len(chunks)} chunks into vector store")
        
        # Add embeddings to vector store
        self.vector_store.add_embeddings(chunks)
        
        # Save to disk
        self.vector_store.save()
        
        logger.info("Document ingestion complete")
    
    def search_documents(self, query_embedding: np.ndarray, top_k: int = None) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        return self.vector_store.search(query_embedding, top_k)
    
    def load_existing_store(self) -> bool:
        """Load existing vector store if available"""
        return self.vector_store.load()
    
    def get_store_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return self.vector_store.get_stats()