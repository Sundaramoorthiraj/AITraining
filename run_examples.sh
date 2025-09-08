#!/bin/bash

# Script to run examples for the Local RAG System

echo "=== Local RAG System Examples ==="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import torch, transformers, sentence_transformers, faiss" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Required packages are not installed"
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi

echo "Dependencies OK"
echo ""

# Create necessary directories
mkdir -p documents vector_store cache

# Run basic examples
echo "Running basic usage examples..."
python3 examples/basic_usage.py

echo ""
echo "Running advanced usage examples..."
python3 examples/advanced_usage.py

echo ""
echo "=== Examples completed! ==="
echo ""
echo "To try the system yourself:"
echo "1. Add your documents to the 'documents/' directory"
echo "2. Run: python3 main.py --mode ingest"
echo "3. Run: python3 main.py --mode interactive"
echo "4. Or start the web interface: streamlit run app.py"