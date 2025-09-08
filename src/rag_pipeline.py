"""
Main RAG pipeline that combines document processing, embeddings, vector store, and LLM inference
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .document_processor import DocumentProcessor
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStoreManager
from .llm_inference import LocalLLMInference
from config import Config

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline for document question-answering"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Initialize components
        self.document_processor = DocumentProcessor(self.config)
        self.embedding_generator = EmbeddingGenerator(self.config)
        self.vector_store_manager = VectorStoreManager(self.config)
        self.llm_inference = LocalLLMInference(self.config)
        
        # State tracking
        self.is_initialized = False
        self.documents_processed = False
    
    def initialize(self) -> None:
        """Initialize the RAG pipeline components"""
        try:
            logger.info("Initializing RAG pipeline...")
            
            # Load embedding model
            self.embedding_generator.load_model()
            
            # Try to load existing vector store
            store_loaded = self.vector_store_manager.load_existing_store()
            if store_loaded:
                logger.info("Loaded existing vector store")
                self.documents_processed = True
            else:
                logger.info("No existing vector store found")
            
            # Load LLM model
            self.llm_inference.load_model()
            
            self.is_initialized = True
            logger.info("RAG pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {e}")
            raise
    
    def ingest_documents(self, documents_path: str = None) -> None:
        """Process documents and create vector store"""
        if documents_path is None:
            documents_path = self.config.documents_path
        
        try:
            logger.info(f"Ingesting documents from: {documents_path}")
            
            # Process documents
            chunks = self.document_processor.process_directory(documents_path)
            
            if not chunks:
                logger.warning("No document chunks were created")
                return
            
            # Generate embeddings
            chunks_with_embeddings = self.embedding_generator.process_chunks(chunks)
            
            # Store in vector database
            self.vector_store_manager.ingest_documents(chunks_with_embeddings)
            
            self.documents_processed = True
            logger.info("Document ingestion completed successfully")
            
        except Exception as e:
            logger.error(f"Error ingesting documents: {e}")
            raise
    
    def query(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """Query the RAG system with a question"""
        if not self.is_initialized:
            self.initialize()
        
        if not self.documents_processed:
            logger.warning("No documents have been processed yet")
            return {
                "answer": "No documents have been processed yet. Please ingest documents first.",
                "context_chunks": [],
                "sources": []
            }
        
        try:
            logger.info(f"Processing query: {question}")
            
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_single_embedding(question)
            
            # Search for relevant chunks
            relevant_chunks = self.vector_store_manager.search_documents(
                query_embedding, top_k
            )
            
            if not relevant_chunks:
                logger.warning("No relevant chunks found for the query")
                return {
                    "answer": "I couldn't find relevant information to answer your question.",
                    "context_chunks": [],
                    "sources": []
                }
            
            # Generate answer using LLM
            answer = self.llm_inference.answer_question(question, relevant_chunks)
            
            # Extract sources
            sources = list(set([chunk['chunk']['source'] for chunk in relevant_chunks]))
            
            result = {
                "answer": answer,
                "context_chunks": [chunk['chunk'] for chunk in relevant_chunks],
                "sources": sources,
                "similarity_scores": [chunk['score'] for chunk in relevant_chunks]
            }
            
            logger.info("Query processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "context_chunks": [],
                "sources": []
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {
            "initialized": self.is_initialized,
            "documents_processed": self.documents_processed,
            "vector_store": self.vector_store_manager.get_store_stats(),
            "llm_model": self.llm_inference.get_model_info()
        }
        
        if self.is_initialized:
            stats["embedding_model"] = {
                "name": self.config.embedding_model,
                "dimension": self.embedding_generator.get_embedding_dimension()
            }
        
        return stats
    
    def clear_vector_store(self) -> None:
        """Clear the vector store and reset document processing status"""
        self.vector_store_manager.vector_store.clear()
        self.documents_processed = False
        logger.info("Vector store cleared")
    
    def add_documents(self, documents_path: str) -> None:
        """Add new documents to existing vector store"""
        try:
            logger.info(f"Adding documents from: {documents_path}")
            
            # Process new documents
            new_chunks = self.document_processor.process_directory(documents_path)
            
            if not new_chunks:
                logger.warning("No new document chunks were created")
                return
            
            # Generate embeddings for new chunks
            new_chunks_with_embeddings = self.embedding_generator.process_chunks(new_chunks)
            
            # Add to existing vector store
            self.vector_store_manager.vector_store.add_embeddings(new_chunks_with_embeddings)
            self.vector_store_manager.vector_store.save()
            
            logger.info("New documents added successfully")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process multiple questions in batch"""
        results = []
        
        for question in questions:
            result = self.query(question)
            results.append({
                "question": question,
                **result
            })
        
        return results