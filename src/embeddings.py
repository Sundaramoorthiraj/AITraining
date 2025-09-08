"""
Embedding generation module using Sentence Transformers
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import torch

from config import Config

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Handles text embedding generation using Sentence Transformers"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.device = "cpu"  # Force CPU usage for consistency
        
    def load_model(self) -> None:
        """Load the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self.model = SentenceTransformer(
                self.config.embedding_model,
                device=self.device,
                cache_folder=self.config.cache_dir
            )
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if self.model is None:
            self.load_model()
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            # Process in batches to manage memory
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                    batch_size=min(batch_size, len(batch_texts))
                )
                all_embeddings.append(batch_embeddings)
            
            embeddings = np.vstack(all_embeddings)
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if self.model is None:
            self.load_model()
        
        try:
            embedding = self.model.encode(
                [text],
                convert_to_numpy=True
            )
            return embedding[0]  # Return single embedding, not batch
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model"""
        if self.model is None:
            self.load_model()
        
        # Generate a test embedding to get dimension
        test_embedding = self.generate_single_embedding("test")
        return len(test_embedding)
    
    def process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process document chunks and add embeddings"""
        if not chunks:
            return []
        
        # Extract texts for embedding generation
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i]
        
        logger.info(f"Added embeddings to {len(chunks)} chunks")
        return chunks


class EmbeddingSimilarity:
    """Utility class for computing embedding similarities"""
    
    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    @staticmethod
    def euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute euclidean distance between two embeddings"""
        return float(np.linalg.norm(embedding1 - embedding2))
    
    @staticmethod
    def find_most_similar(query_embedding: np.ndarray, 
                         candidate_embeddings: List[np.ndarray],
                         top_k: int = 5) -> List[tuple]:
        """Find most similar embeddings to query"""
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = EmbeddingSimilarity.cosine_similarity(query_embedding, candidate)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]