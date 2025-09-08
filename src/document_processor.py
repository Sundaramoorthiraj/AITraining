"""
Document processing module for extracting and chunking text from various file formats
"""
import os
import logging
from typing import List, Dict, Any
from pathlib import Path

import PyPDF2
from docx import Document
import openpyxl
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document parsing and text extraction from various formats"""
    
    def __init__(self, config: Config):
        self.config = config
        self.supported_formats = config.supported_formats
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            return ""
    
    def extract_text_from_excel(self, file_path: str) -> str:
        """Extract text from Excel file"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            text = ""
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                text += f"Sheet: {sheet_name}\n"
                text += df.to_string() + "\n\n"
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from Excel {file_path}: {e}")
            return ""
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            logger.error(f"Error extracting text from TXT {file_path}: {e}")
            return ""
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from file based on extension"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            return self.extract_text_from_docx(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            return self.extract_text_from_excel(file_path)
        elif file_ext == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_ext}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks"""
        if chunk_size is None:
            chunk_size = self.config.chunk_size
        if overlap is None:
            overlap = self.config.chunk_overlap
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start + chunk_size - 100, start)
                sentence_endings = ['.', '!', '?', '\n\n']
                
                for i in range(end - 1, search_start, -1):
                    if text[i] in sentence_endings:
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a single document and return chunks with metadata"""
        logger.info(f"Processing document: {file_path}")
        
        text = self.extract_text(file_path)
        if not text:
            logger.warning(f"No text extracted from {file_path}")
            return []
        
        chunks = self.chunk_text(text)
        
        # Create chunk metadata
        document_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                'text': chunk,
                'source': file_path,
                'chunk_id': f"{Path(file_path).stem}_{i}",
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
            document_chunks.append(chunk_data)
        
        logger.info(f"Created {len(chunks)} chunks from {file_path}")
        return document_chunks
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Process all supported documents in a directory"""
        all_chunks = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return all_chunks
        
        # Find all supported files
        supported_files = []
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                supported_files.append(file_path)
        
        logger.info(f"Found {len(supported_files)} supported files in {directory_path}")
        
        # Process each file
        for file_path in supported_files:
            try:
                chunks = self.process_document(str(file_path))
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks