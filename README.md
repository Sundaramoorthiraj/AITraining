# Resume Parser Application

A comprehensive resume parsing application that extracts key entities from resumes in multiple formats including PDF, DOC, DOCX, RTF, and image files (JPG, PNG).

## Features

- **Multi-format Support**: Handles PDF, DOC, DOCX, RTF, JPG, JPEG, and PNG files
- **Entity Extraction**: Extracts the following information:
  - Personal Information (First Name, Last Name)
  - Contact Details (Email, Phone Number)
  - Professional Links (LinkedIn, GitHub)
  - Education Details
  - Professional Experience
  - Project Information
- **Web Interface**: User-friendly web interface for file upload and results display
- **API Endpoints**: RESTful API for programmatic access
- **OCR Support**: Text extraction from image files using Tesseract OCR
- **NLP Processing**: Uses spaCy for intelligent text processing and entity recognition

## Installation

### Prerequisites

- Python 3.7 or higher
- Tesseract OCR (for image processing)

### Install Tesseract OCR

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

### Setup Application

1. Clone or download the application files
2. Run the setup script:
```bash
python setup.py
```

Or manually install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

### Web Interface

1. Start the application:
```bash
python app.py
```

2. Open your browser and navigate to: `http://localhost:5000`

3. Upload a resume file using the drag-and-drop interface or file browser

4. View the extracted information in a structured format

### API Usage

#### Upload and Parse Resume

**Endpoint:** `POST /api/parse`

**Request:**
- Content-Type: `multipart/form-data`
- Body: File upload with key `file`

**Response:**
```json
{
  "file_name": "resume.pdf",
  "file_type": "application/pdf",
  "names": {
    "first_name": "John",
    "last_name": "Doe"
  },
  "emails": ["john.doe@email.com"],
  "phones": ["+1 (555) 123-4567"],
  "linkedin": ["https://linkedin.com/in/johndoe"],
  "github": ["https://github.com/johndoe"],
  "education": [
    "Bachelor of Science in Computer Science",
    "University of Technology, 2020"
  ],
  "experience": [
    "Software Engineer at Tech Corp (2020-2023)",
    "Developed web applications using Python and React"
  ],
  "projects": [
    "E-commerce Platform - Built with Django and PostgreSQL",
    "Mobile App - React Native application for task management"
  ],
  "raw_text": "Extracted text content..."
}
```

#### Example API Call

```bash
curl -X POST -F "file=@resume.pdf" http://localhost:5000/api/parse
```

## Supported File Formats

| Format | Extension | Processing Method |
|--------|-----------|-------------------|
| PDF | .pdf | PyPDF2 |
| Word Document | .doc, .docx | python-docx |
| Rich Text Format | .rtf | python-rtf |
| Images | .jpg, .jpeg, .png | Tesseract OCR |

## Configuration

### File Size Limits
- Maximum file size: 16MB
- Configurable in `app.py`

### Supported Extensions
- Configurable in `app.py` in the `ALLOWED_EXTENSIONS` set

## Architecture

### Core Components

1. **Flask Web Application** (`app.py`)
   - Handles file uploads
   - Serves web interface
   - Provides API endpoints

2. **Resume Parser** (`resume_parser.py`)
   - Text extraction from various formats
   - Entity extraction using regex and NLP
   - Structured data output

3. **Web Interface** (`templates/index.html`)
   - Drag-and-drop file upload
   - Results display
   - Responsive design

### Text Processing Pipeline

1. **File Type Detection**: Uses python-magic to detect file format
2. **Text Extraction**: Format-specific extraction methods
3. **Entity Recognition**: 
   - Regex patterns for structured data (email, phone, URLs)
   - spaCy NLP for names and general text processing
   - Keyword-based section identification
4. **Data Structuring**: Organizes extracted information into structured format

## Error Handling

The application includes comprehensive error handling for:
- Unsupported file formats
- File size limits
- Text extraction failures
- Invalid file content
- Network errors

## Performance Considerations

- Files are processed in memory and cleaned up after processing
- OCR processing may take longer for large image files
- Consider implementing caching for frequently processed files
- For production use, consider implementing file size limits and rate limiting

## Security Features

- File type validation
- File size limits
- Secure filename handling
- Temporary file cleanup

## Troubleshooting

### Common Issues

1. **spaCy model not found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **Tesseract not found**
   - Install Tesseract OCR on your system
   - Ensure it's in your system PATH

3. **Permission errors**
   - Ensure the application has write permissions for the uploads directory

4. **Memory issues with large files**
   - Reduce file size or implement streaming processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the error messages in the application logs
3. Ensure all dependencies are properly installed
4. Verify file format compatibility