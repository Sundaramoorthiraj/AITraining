#!/usr/bin/env python3
"""
Advanced usage examples for the Local RAG System
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import RAGPipeline
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStoreManager
from src.llm_inference import LocalLLMInference
from config import Config

def example_component_usage():
    """Example of using individual components"""
    print("=== Individual Component Usage ===")
    
    config = Config()
    
    # Document processing
    print("1. Document Processing")
    processor = DocumentProcessor(config)
    chunks = processor.process_directory("./documents/")
    print(f"   Created {len(chunks)} chunks")
    
    # Embedding generation
    print("2. Embedding Generation")
    embedding_gen = EmbeddingGenerator(config)
    chunks_with_embeddings = embedding_gen.process_chunks(chunks)
    print(f"   Generated embeddings for {len(chunks_with_embeddings)} chunks")
    
    # Vector store management
    print("3. Vector Store Management")
    vector_manager = VectorStoreManager(config)
    vector_manager.ingest_documents(chunks_with_embeddings)
    print(f"   Stored {len(chunks_with_embeddings)} embeddings in vector store")
    
    # LLM inference
    print("4. LLM Inference")
    llm = LocalLLMInference(config)
    query = "What is the Local RAG System?"
    query_embedding = embedding_gen.generate_single_embedding(query)
    relevant_chunks = vector_manager.search_documents(query_embedding)
    answer = llm.answer_question(query, relevant_chunks)
    print(f"   Query: {query}")
    print(f"   Answer: {answer}")

def example_custom_chunking():
    """Example with custom chunking strategy"""
    print("\n=== Custom Chunking Strategy ===")
    
    config = Config()
    config.chunk_size = 200  # Very small chunks
    config.chunk_overlap = 50  # High overlap
    
    processor = DocumentProcessor(config)
    chunks = processor.process_directory("./documents/")
    
    print(f"Created {len(chunks)} chunks with custom settings:")
    print(f"  Chunk size: {config.chunk_size}")
    print(f"  Chunk overlap: {config.chunk_overlap}")
    
    # Show first few chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"  Source: {chunk['source']}")
        print(f"  Text: {chunk['text'][:100]}...")

def example_embedding_analysis():
    """Example of embedding analysis"""
    print("\n=== Embedding Analysis ===")
    
    config = Config()
    embedding_gen = EmbeddingGenerator(config)
    
    # Generate embeddings for sample texts
    texts = [
        "The Local RAG System processes documents",
        "Vector embeddings enable semantic search",
        "FAISS provides efficient similarity search",
        "Local LLM inference ensures privacy"
    ]
    
    embeddings = embedding_gen.generate_embeddings(texts)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Compute similarities
    from src.embeddings import EmbeddingSimilarity
    
    query_embedding = embeddings[0]  # First text as query
    similarities = []
    
    for i, embedding in enumerate(embeddings):
        similarity = EmbeddingSimilarity.cosine_similarity(query_embedding, embedding)
        similarities.append((texts[i], similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print("\nSimilarity scores:")
    for text, score in similarities:
        print(f"  {score:.3f}: {text}")

def example_vector_store_operations():
    """Example of vector store operations"""
    print("\n=== Vector Store Operations ===")
    
    config = Config()
    vector_manager = VectorStoreManager(config)
    
    # Load existing store
    loaded = vector_manager.load_existing_store()
    if loaded:
        stats = vector_manager.get_store_stats()
        print(f"Loaded vector store with {stats['total_embeddings']} embeddings")
        
        # Search example
        embedding_gen = EmbeddingGenerator(config)
        query = "What are the key features?"
        query_embedding = embedding_gen.generate_single_embedding(query)
        
        results = vector_manager.search_documents(query_embedding, top_k=3)
        print(f"\nSearch results for: '{query}'")
        for i, result in enumerate(results, 1):
            chunk = result['chunk']
            print(f"  {i}. Score: {result['score']:.3f}")
            print(f"     Source: {chunk['source']}")
            print(f"     Text: {chunk['text'][:100]}...")
    else:
        print("No existing vector store found")

def example_llm_customization():
    """Example of LLM customization"""
    print("\n=== LLM Customization ===")
    
    config = Config()
    config.temperature = 0.3  # More deterministic
    config.max_tokens = 256   # Shorter responses
    config.top_p = 0.8       # More focused sampling
    
    llm = LocalLLMInference(config)
    
    # Test different parameters
    prompt = "Explain the benefits of local LLM inference:"
    
    print("Testing different LLM parameters:")
    
    # Low temperature (deterministic)
    response1 = llm.generate_response(prompt, temperature=0.1)
    print(f"\nLow temperature (0.1): {response1[:100]}...")
    
    # High temperature (creative)
    response2 = llm.generate_response(prompt, temperature=0.9)
    print(f"\nHigh temperature (0.9): {response2[:100]}...")
    
    # Different top_p values
    response3 = llm.generate_response(prompt, top_p=0.5)
    print(f"\nLow top_p (0.5): {response3[:100]}...")

def example_error_handling():
    """Example of error handling"""
    print("\n=== Error Handling ===")
    
    config = Config()
    rag = RAGPipeline(config)
    
    # Test with no documents
    print("Testing query without documents:")
    result = rag.query("What is the system?")
    print(f"Result: {result['answer']}")
    
    # Test with invalid path
    print("\nTesting with invalid document path:")
    try:
        rag.ingest_documents("/nonexistent/path")
    except Exception as e:
        print(f"Expected error: {e}")
    
    # Test with empty query
    print("\nTesting with empty query:")
    result = rag.query("")
    print(f"Result: {result['answer']}")

if __name__ == "__main__":
    try:
        example_component_usage()
        example_custom_chunking()
        example_embedding_analysis()
        example_vector_store_operations()
        example_llm_customization()
        example_error_handling()
        
        print("\n=== All advanced examples completed successfully! ===")
        
    except Exception as e:
        print(f"Error running advanced examples: {e}")
        sys.exit(1)