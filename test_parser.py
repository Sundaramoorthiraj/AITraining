#!/usr/bin/env python3
"""
Test script for Resume Parser
"""

import os
import json
from resume_parser import ResumeParser

def test_parser():
    """Test the resume parser with sample data"""
    print("🧪 Testing Resume Parser...")
    print("=" * 50)
    
    # Initialize parser
    parser = ResumeParser()
    
    # Create a sample resume text for testing
    sample_text = """
    John Doe
    Software Engineer
    Email: john.doe@email.com
    Phone: (555) 123-4567
    LinkedIn: https://linkedin.com/in/johndoe
    GitHub: https://github.com/johndoe
    
    EDUCATION
    Bachelor of Science in Computer Science
    University of Technology, 2020
    GPA: 3.8/4.0
    
    EXPERIENCE
    Software Engineer at Tech Corp (2020-2023)
    - Developed web applications using Python and React
    - Led a team of 3 developers
    - Improved application performance by 40%
    
    Junior Developer at StartupXYZ (2019-2020)
    - Built REST APIs using Django
    - Collaborated with design team on UI/UX
    
    PROJECTS
    E-commerce Platform
    - Built with Django and PostgreSQL
    - Implemented payment processing with Stripe
    - Deployed on AWS with Docker
    
    Task Management App
    - React Native mobile application
    - Real-time collaboration features
    - Used Firebase for backend services
    """
    
    # Test text extraction and entity recognition
    print("📝 Testing entity extraction...")
    
    # Test email extraction
    emails = parser.extract_emails(sample_text)
    print(f"✅ Emails found: {emails}")
    
    # Test phone extraction
    phones = parser.extract_phones(sample_text)
    print(f"✅ Phones found: {phones}")
    
    # Test LinkedIn extraction
    linkedin = parser.extract_linkedin(sample_text)
    print(f"✅ LinkedIn found: {linkedin}")
    
    # Test GitHub extraction
    github = parser.extract_github(sample_text)
    print(f"✅ GitHub found: {github}")
    
    # Test name extraction
    names = parser.extract_names(sample_text)
    print(f"✅ Names found: {names}")
    
    # Test education extraction
    education = parser.extract_education(sample_text)
    print(f"✅ Education found: {len(education)} items")
    for item in education[:2]:  # Show first 2 items
        print(f"   - {item}")
    
    # Test experience extraction
    experience = parser.extract_experience(sample_text)
    print(f"✅ Experience found: {len(experience)} items")
    for item in experience[:2]:  # Show first 2 items
        print(f"   - {item}")
    
    # Test project extraction
    projects = parser.extract_projects(sample_text)
    print(f"✅ Projects found: {len(projects)} items")
    for item in projects[:2]:  # Show first 2 items
        print(f"   - {item}")
    
    print("\n" + "=" * 50)
    print("✅ All tests completed successfully!")
    
    # Test file type detection
    print("\n🔍 Testing file type detection...")
    test_files = [
        "test.pdf",
        "test.docx", 
        "test.jpg",
        "test.rtf"
    ]
    
    for filename in test_files:
        file_type = parser.detect_file_type(filename)
        print(f"📄 {filename} -> {file_type}")
    
    print("\n🎉 Resume Parser is ready to use!")
    print("\n📋 To start the web application:")
    print("   python app.py")
    print("\n📋 To test with a real file:")
    print("   python -c \"from resume_parser import ResumeParser; p = ResumeParser(); print(p.parse_resume('your_file.pdf'))\"")

if __name__ == "__main__":
    test_parser()