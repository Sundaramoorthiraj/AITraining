#!/usr/bin/env python3
"""
Basic usage examples for the Local RAG System
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import RAGPipeline
from config import Config

def example_basic_usage():
    """Basic usage example"""
    print("=== Basic RAG System Usage ===")
    
    # Initialize the RAG pipeline
    config = Config()
    rag = RAGPipeline(config)
    
    # Process documents
    print("Processing documents...")
    rag.ingest_documents("./documents/")
    
    # Ask questions
    questions = [
        "What is the Local RAG System?",
        "What are the key features?",
        "What models are used for embeddings?",
        "What are the performance considerations?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = rag.query(question)
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")

def example_custom_configuration():
    """Example with custom configuration"""
    print("\n=== Custom Configuration Example ===")
    
    # Create custom configuration
    config = Config()
    config.chunk_size = 256  # Smaller chunks
    config.chunk_overlap = 25
    config.top_k_results = 3
    config.temperature = 0.5  # More deterministic responses
    
    # Initialize with custom config
    rag = RAGPipeline(config)
    
    # Process documents
    rag.ingest_documents("./documents/")
    
    # Ask a question
    question = "What are the security benefits?"
    result = rag.query(question)
    
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")

def example_batch_processing():
    """Example of batch question processing"""
    print("\n=== Batch Processing Example ===")
    
    config = Config()
    rag = RAGPipeline(config)
    
    # Process documents
    rag.ingest_documents("./documents/")
    
    # Batch questions
    questions = [
        "What file formats are supported?",
        "How does the vector store work?",
        "What are the installation requirements?",
        "What are the future enhancements?"
    ]
    
    results = rag.batch_query(questions)
    
    for result in results:
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}")

def example_system_statistics():
    """Example of getting system statistics"""
    print("\n=== System Statistics Example ===")
    
    config = Config()
    rag = RAGPipeline(config)
    
    # Process documents
    rag.ingest_documents("./documents/")
    
    # Get system stats
    stats = rag.get_system_stats()
    
    print("System Statistics:")
    print(f"  Initialized: {stats['initialized']}")
    print(f"  Documents processed: {stats['documents_processed']}")
    print(f"  Vector store status: {stats['vector_store']['status']}")
    print(f"  Total embeddings: {stats['vector_store']['total_embeddings']}")
    print(f"  Total chunks: {stats['vector_store']['total_chunks']}")
    print(f"  LLM model: {stats['llm_model']['model_name']}")

if __name__ == "__main__":
    try:
        example_basic_usage()
        example_custom_configuration()
        example_batch_processing()
        example_system_statistics()
        
        print("\n=== All examples completed successfully! ===")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        sys.exit(1)