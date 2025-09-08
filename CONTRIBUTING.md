# Contributing to Local RAG System

Thank you for your interest in contributing to the Local RAG System! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a new branch for your feature or bugfix
4. Install dependencies: `pip install -r requirements.txt`
5. Make your changes
6. Test your changes
7. Submit a pull request

## Development Setup

### Prerequisites
- Python 3.8 or higher
- Git
- 8GB+ RAM recommended

### Installation
```bash
git clone <your-fork-url>
cd local-rag-system
pip install -r requirements.txt
```

### Running Tests
```bash
# Run basic examples
python examples/basic_usage.py

# Run advanced examples
python examples/advanced_usage.py

# Run the full system
python main.py --mode stats
```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Keep functions focused and small
- Use meaningful variable and function names

## Testing

Before submitting a pull request:

1. Run the basic examples to ensure core functionality works
2. Test with different document types (PDF, DOCX, TXT)
3. Verify that the web interface works correctly
4. Check that error handling works as expected

## Areas for Contribution

### High Priority
- Additional document format support (PPTX, RTF, etc.)
- Better error handling and user feedback
- Performance optimizations
- More quantized model support

### Medium Priority
- Multi-language support
- Advanced chunking strategies
- Model fine-tuning capabilities
- Distributed processing support

### Low Priority
- Additional UI frameworks (Gradio, etc.)
- Docker containerization
- CI/CD pipeline improvements
- Documentation improvements

## Bug Reports

When reporting bugs, please include:

1. Python version
2. Operating system
3. Steps to reproduce
4. Expected vs actual behavior
5. Error messages (if any)
6. System specifications (RAM, CPU)

## Feature Requests

When requesting features, please include:

1. Use case description
2. Proposed implementation approach
3. Potential impact on existing functionality
4. Any relevant examples or references

## Pull Request Process

1. Ensure your code follows the style guidelines
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all examples still work
5. Write a clear description of your changes

## Code Review

All submissions require review. Please:

- Respond to feedback promptly
- Make requested changes
- Test changes thoroughly
- Keep pull requests focused and small

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## Questions?

Feel free to open an issue for questions about contributing or the project in general.