import os
import re
import json
import magic
import spacy
# import pandas as pd  # Not needed for this implementation
import PyPDF2
import pytesseract
from PIL import Image
import cv2
import numpy as np
from docx import Document
import phonenumbers
from email_validator import validate_email, EmailNotValidError
import requests
from io import BytesIO
# import rtf  # Not available, will use alternative method

class ResumeParser:
    def __init__(self):
        # Load spaCy model for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Please install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Regex patterns for entity extraction
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            'linkedin': r'(?:https?://)?(?:www\.)?linkedin\.com/in/([a-zA-Z0-9-]+)',
            'github': r'(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9-]+)',
            'url': r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
        }
        
        # Education keywords
        self.education_keywords = [
            'education', 'academic', 'degree', 'bachelor', 'master', 'phd', 'doctorate',
            'university', 'college', 'institute', 'school', 'diploma', 'certificate',
            'gpa', 'grade', 'graduation', 'graduated', 'major', 'minor', 'field of study'
        ]
        
        # Experience keywords
        self.experience_keywords = [
            'experience', 'employment', 'work history', 'career', 'professional',
            'position', 'role', 'job', 'employment', 'worked', 'worked as',
            'responsibilities', 'achievements', 'projects', 'skills'
        ]
        
        # Project keywords
        self.project_keywords = [
            'project', 'projects', 'portfolio', 'developed', 'built', 'created',
            'designed', 'implemented', 'contributed', 'collaborated'
        ]

    def detect_file_type(self, filepath):
        """Detect file type using python-magic"""
        try:
            mime_type = magic.from_file(filepath, mime=True)
            return mime_type
        except:
            # Fallback to file extension
            ext = os.path.splitext(filepath)[1].lower()
            mime_map = {
                '.pdf': 'application/pdf',
                '.doc': 'application/msword',
                '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                '.rtf': 'application/rtf',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.txt': 'text/plain'
            }
            return mime_map.get(ext, 'unknown')

    def extract_text_from_pdf(self, filepath):
        """Extract text from PDF files"""
        text = ""
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
        return text

    def extract_text_from_docx(self, filepath):
        """Extract text from DOCX files"""
        text = ""
        try:
            doc = Document(filepath)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            print(f"Error extracting DOCX text: {e}")
        return text

    def extract_text_from_rtf(self, filepath):
        """Extract text from RTF files using simple regex"""
        text = ""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                rtf_content = file.read()
                # Simple RTF text extraction - remove RTF control codes
                import re
                # Remove RTF control words and braces
                text = re.sub(r'\\[a-z]+\d*\s?', '', rtf_content)
                text = re.sub(r'[{}]', '', text)
                # Clean up extra whitespace
                text = re.sub(r'\s+', ' ', text).strip()
        except Exception as e:
            print(f"Error extracting RTF text: {e}")
        return text

    def extract_text_from_image(self, filepath):
        """Extract text from image files using OCR"""
        text = ""
        try:
            # Load image
            image = cv2.imread(filepath)
            
            # Preprocess image for better OCR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get binary image
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Use pytesseract for OCR
            text = pytesseract.image_to_string(thresh)
            
        except Exception as e:
            print(f"Error extracting text from image: {e}")
        return text

    def extract_text_from_txt(self, filepath):
        """Extract text from TXT files"""
        text = ""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
        except Exception as e:
            print(f"Error extracting TXT text: {e}")
        return text

    def extract_text(self, filepath):
        """Extract text from various file formats"""
        file_type = self.detect_file_type(filepath)
        
        if file_type == 'application/pdf':
            return self.extract_text_from_pdf(filepath)
        elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            return self.extract_text_from_docx(filepath)
        elif file_type == 'application/rtf':
            return self.extract_text_from_rtf(filepath)
        elif file_type in ['image/jpeg', 'image/png']:
            return self.extract_text_from_image(filepath)
        elif file_type == 'text/plain':
            return self.extract_text_from_txt(filepath)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def extract_emails(self, text):
        """Extract email addresses from text"""
        emails = re.findall(self.patterns['email'], text)
        valid_emails = []
        for email in emails:
            try:
                validate_email(email)
                valid_emails.append(email)
            except EmailNotValidError:
                continue
        return list(set(valid_emails))

    def extract_phones(self, text):
        """Extract phone numbers from text"""
        phone_matches = re.findall(self.patterns['phone'], text)
        phones = []
        for match in phone_matches:
            # Reconstruct phone number
            phone = ''.join(match)
            if len(phone) >= 10:
                try:
                    # Parse and format phone number
                    parsed = phonenumbers.parse(phone, "US")
                    if phonenumbers.is_valid_number(parsed):
                        formatted = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.NATIONAL)
                        phones.append(formatted)
                except:
                    # If parsing fails, add the raw match
                    phones.append(phone)
        return list(set(phones))

    def extract_linkedin(self, text):
        """Extract LinkedIn profile URLs"""
        linkedin_matches = re.findall(self.patterns['linkedin'], text)
        return [f"https://linkedin.com/in/{match}" for match in linkedin_matches]

    def extract_github(self, text):
        """Extract GitHub profile URLs"""
        github_matches = re.findall(self.patterns['github'], text)
        return [f"https://github.com/{match}" for match in github_matches]

    def extract_names(self, text):
        """Extract first and last names using NLP"""
        if not self.nlp:
            return {"first_name": "", "last_name": ""}
        
        doc = self.nlp(text)
        names = []
        
        # Extract PERSON entities
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                names.append(ent.text)
        
        # If no PERSON entities found, try to extract from the beginning of the text
        if not names:
            lines = text.split('\n')
            for line in lines[:5]:  # Check first 5 lines
                line = line.strip()
                if line and len(line.split()) <= 3:  # Likely a name line
                    names.append(line)
        
        if names:
            # Take the first name found
            full_name = names[0].split()
            first_name = full_name[0] if full_name else ""
            last_name = " ".join(full_name[1:]) if len(full_name) > 1 else ""
            return {"first_name": first_name, "last_name": last_name}
        
        return {"first_name": "", "last_name": ""}

    def extract_education(self, text):
        """Extract education details"""
        education_section = []
        lines = text.split('\n')
        
        # Find education section
        education_start = -1
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in self.education_keywords):
                education_start = i
                break
        
        if education_start != -1:
            # Extract education details from the section
            for i in range(education_start, min(education_start + 10, len(lines))):
                line = lines[i].strip()
                if line and len(line) > 10:  # Filter out very short lines
                    education_section.append(line)
        
        return education_section

    def extract_experience(self, text):
        """Extract professional experience"""
        experience_section = []
        lines = text.split('\n')
        
        # Find experience section
        experience_start = -1
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in self.experience_keywords):
                experience_start = i
                break
        
        if experience_start != -1:
            # Extract experience details from the section
            for i in range(experience_start, min(experience_start + 20, len(lines))):
                line = lines[i].strip()
                if line and len(line) > 10:  # Filter out very short lines
                    experience_section.append(line)
        
        return experience_section

    def extract_projects(self, text):
        """Extract project details"""
        project_section = []
        lines = text.split('\n')
        
        # Find project section
        project_start = -1
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in self.project_keywords):
                project_start = i
                break
        
        if project_start != -1:
            # Extract project details from the section
            for i in range(project_start, min(project_start + 15, len(lines))):
                line = lines[i].strip()
                if line and len(line) > 10:  # Filter out very short lines
                    project_section.append(line)
        
        return project_section

    def parse_resume(self, filepath):
        """Main method to parse resume and extract all entities"""
        try:
            # Extract text from file
            text = self.extract_text(filepath)
            
            if not text.strip():
                return {"error": "No text could be extracted from the file"}
            
            # Extract entities
            result = {
                "file_name": os.path.basename(filepath),
                "file_type": self.detect_file_type(filepath),
                "names": self.extract_names(text),
                "emails": self.extract_emails(text),
                "phones": self.extract_phones(text),
                "linkedin": self.extract_linkedin(text),
                "github": self.extract_github(text),
                "education": self.extract_education(text),
                "experience": self.extract_experience(text),
                "projects": self.extract_projects(text),
                "raw_text": text[:1000] + "..." if len(text) > 1000 else text  # Limit raw text for response
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Error parsing resume: {str(e)}"}

if __name__ == "__main__":
    # Test the parser
    parser = ResumeParser()
    print("Resume Parser initialized successfully!")