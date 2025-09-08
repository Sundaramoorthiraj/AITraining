#!/usr/bin/env python3
"""
Main command-line interface for the Local RAG System
"""
import argparse
import logging
import sys
from pathlib import Path

from src.rag_pipeline import RAGPipeline
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories(config: Config) -> None:
    """Ensure required directories exist"""
    Path(config.documents_path).mkdir(exist_ok=True)
    Path(config.vector_store_path).mkdir(exist_ok=True)
    Path(config.cache_dir).mkdir(exist_ok=True)


def ingest_mode(args, config: Config) -> None:
    """Handle document ingestion mode"""
    logger.info("Starting document ingestion...")
    
    # Setup directories
    setup_directories(config)
    
    # Initialize RAG pipeline
    rag = RAGPipeline(config)
    
    # Ingest documents
    documents_path = args.documents or config.documents_path
    rag.ingest_documents(documents_path)
    
    # Print statistics
    stats = rag.get_system_stats()
    print(f"\nIngestion complete!")
    print(f"Vector store contains {stats['vector_store']['total_embeddings']} embeddings")
    print(f"Total chunks processed: {stats['vector_store']['total_chunks']}")


def query_mode(args, config: Config) -> None:
    """Handle query mode"""
    logger.info("Starting query processing...")
    
    # Initialize RAG pipeline
    rag = RAGPipeline(config)
    
    # Process query
    result = rag.query(args.question, top_k=args.top_k)
    
    # Display results
    print(f"\nQuestion: {args.question}")
    print(f"\nAnswer: {result['answer']}")
    
    if result['sources']:
        print(f"\nSources:")
        for source in result['sources']:
            print(f"  - {source}")
    
    if args.verbose and result['context_chunks']:
        print(f"\nContext chunks used:")
        for i, chunk in enumerate(result['context_chunks'], 1):
            print(f"\n{i}. Score: {result['similarity_scores'][i-1]:.3f}")
            print(f"   Source: {chunk['source']}")
            print(f"   Text: {chunk['text'][:200]}...")


def interactive_mode(args, config: Config) -> None:
    """Handle interactive mode"""
    logger.info("Starting interactive mode...")
    
    # Initialize RAG pipeline
    rag = RAGPipeline(config)
    
    print("\n=== Local RAG System - Interactive Mode ===")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'stats' to see system statistics")
    print("Type 'help' for more commands")
    
    while True:
        try:
            question = input("\nEnter your question: ").strip()
            
            if question.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            elif question.lower() == 'stats':
                stats = rag.get_system_stats()
                print(f"\nSystem Statistics:")
                print(f"  Initialized: {stats['initialized']}")
                print(f"  Documents processed: {stats['documents_processed']}")
                if stats['vector_store']['status'] != 'empty':
                    print(f"  Total embeddings: {stats['vector_store']['total_embeddings']}")
                    print(f"  Total chunks: {stats['vector_store']['total_chunks']}")
                continue
            elif question.lower() == 'help':
                print("\nAvailable commands:")
                print("  quit/exit - Exit the program")
                print("  stats - Show system statistics")
                print("  help - Show this help message")
                continue
            elif not question:
                continue
            
            # Process the question
            result = rag.query(question, top_k=args.top_k)
            
            print(f"\nAnswer: {result['answer']}")
            
            if result['sources'] and args.verbose:
                print(f"\nSources: {', '.join(result['sources'])}")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
            print(f"Error: {e}")


def stats_mode(args, config: Config) -> None:
    """Handle stats mode"""
    rag = RAGPipeline(config)
    stats = rag.get_system_stats()
    
    print("\n=== System Statistics ===")
    print(f"Initialized: {stats['initialized']}")
    print(f"Documents processed: {stats['documents_processed']}")
    
    if stats['vector_store']['status'] != 'empty':
        print(f"\nVector Store:")
        print(f"  Total embeddings: {stats['vector_store']['total_embeddings']}")
        print(f"  Embedding dimension: {stats['vector_store']['embedding_dimension']}")
        print(f"  Total chunks: {stats['vector_store']['total_chunks']}")
    
    if stats['llm_model']['status'] == 'loaded':
        print(f"\nLLM Model:")
        print(f"  Model: {stats['llm_model']['model_name']}")
        print(f"  Parameters: {stats['llm_model']['num_parameters']:,}")
        print(f"  Size: {stats['llm_model']['model_size_mb']:.1f} MB")
    
    if 'embedding_model' in stats:
        print(f"\nEmbedding Model:")
        print(f"  Model: {stats['embedding_model']['name']}")
        print(f"  Dimension: {stats['embedding_model']['dimension']}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Local RAG System for Document Q&A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest documents
  python main.py --mode ingest --documents ./my_documents/
  
  # Ask a single question
  python main.py --mode query --question "What is the main topic?"
  
  # Interactive mode
  python main.py --mode interactive
  
  # Show system statistics
  python main.py --mode stats
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['ingest', 'query', 'interactive', 'stats'],
        required=True,
        help='Operation mode'
    )
    
    parser.add_argument(
        '--documents',
        type=str,
        help='Path to documents directory (for ingest mode)'
    )
    
    parser.add_argument(
        '--question',
        type=str,
        help='Question to ask (for query mode)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top results to retrieve (default: 5)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output including sources and context'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom config file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    if args.config:
        # TODO: Implement config file loading
        pass
    
    try:
        if args.mode == 'ingest':
            ingest_mode(args, config)
        elif args.mode == 'query':
            if not args.question:
                print("Error: --question is required for query mode")
                sys.exit(1)
            query_mode(args, config)
        elif args.mode == 'interactive':
            interactive_mode(args, config)
        elif args.mode == 'stats':
            stats_mode(args, config)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()