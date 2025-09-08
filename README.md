# Local RAG System for Document Q&A

A complete Retrieval-Augmented Generation (RAG) system that runs quantized open-source LLMs on CPU for document question-and-answer tasks. This project enables private, cost-effective document processing without relying on third-party LLM providers.

## Features

- **Document Processing**: Support for PDF, DOC, XLS, and TXT files
- **Vector Embeddings**: Uses SBERT all-MiniLM-L6-v2 for semantic search
- **Vector Store**: FAISS for efficient similarity search
- **Local LLM**: Quantized Llama 2 for CPU inference
- **Web Interface**: Streamlit-based UI for easy interaction
- **Privacy-First**: All processing happens locally

## Architecture

The system follows a three-stage pipeline:

1. **Document Ingestion**: Extract and chunk text from various document formats
2. **Vector Storage**: Generate embeddings and store in FAISS vector database
3. **Query Processing**: Retrieve relevant chunks and generate answers using local LLM

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd local-rag-system

# Install dependencies
pip install -r requirements.txt

# Download the quantized Llama 2 model (optional, will download on first use)
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')"
```

## Quick Start

### Command Line Interface

```bash
# Process documents and create vector store
python main.py --mode ingest --documents ./documents/

# Ask questions
python main.py --mode query --question "What is the main topic of the documents?"
```

### Web Interface

```bash
# Launch Streamlit interface
streamlit run app.py
```

## Configuration

Edit `config.py` to customize:

- Model settings (embedding model, LLM model)
- Chunk size and overlap
- Vector store parameters
- Inference settings

## Project Structure

```
├── main.py                 # Main CLI interface
├── app.py                  # Streamlit web interface
├── config.py              # Configuration settings
├── src/
│   ├── document_processor.py    # Document parsing and chunking
│   ├── embeddings.py            # Vector embedding generation
│   ├── vector_store.py          # FAISS vector database
│   ├── llm_inference.py         # Local LLM inference
│   └── rag_pipeline.py          # Complete RAG pipeline
├── documents/              # Place your documents here
├── vector_store/           # Generated vector database
└── requirements.txt        # Python dependencies
```

## Usage Examples

### Processing Documents

```python
from src.rag_pipeline import RAGPipeline

# Initialize the pipeline
rag = RAGPipeline()

# Process documents
rag.ingest_documents("./documents/")

# Ask questions
answer = rag.query("What are the key findings?")
print(answer)
```

### Custom Configuration

```python
from src.rag_pipeline import RAGPipeline
from config import Config

# Custom configuration
config = Config()
config.chunk_size = 512
config.chunk_overlap = 50
config.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

rag = RAGPipeline(config)
```

## Performance Notes

- **CPU Inference**: Optimized for CPU-only environments
- **Memory Usage**: Quantized models reduce memory requirements
- **Speed**: Trade-off between speed and accuracy for cost-effective deployment
- **Scalability**: Vector store supports thousands of documents

## Troubleshooting

### Common Issues

1. **Model Download**: First run may take time to download models
2. **Memory**: Ensure sufficient RAM for model loading
3. **Dependencies**: Use exact versions in requirements.txt

### Performance Optimization

- Adjust chunk size based on document types
- Use smaller embedding models for faster processing
- Consider model quantization levels

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.