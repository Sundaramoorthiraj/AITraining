#!/usr/bin/env python3
"""
Setup script for Resume Parser Application
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def main():
    print("🚀 Setting up Resume Parser Application...")
    print("=" * 50)
    
    # Install Python dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("❌ Failed to install dependencies. Please check your Python environment.")
        return False
    
    # Download spaCy model
    if not run_command("python -m spacy download en_core_web_sm", "Downloading spaCy English model"):
        print("❌ Failed to download spaCy model. You may need to install it manually.")
        print("   Try: python -m spacy download en_core_web_sm")
    
    # Create necessary directories
    directories = ['uploads', 'templates', 'static']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run the application: python app.py")
    print("2. Open your browser and go to: http://localhost:5000")
    print("3. Upload a resume file to test the parser")
    print("\n🔧 API Usage:")
    print("POST /api/parse - Upload and parse resume programmatically")
    print("POST /upload - Upload and parse resume via web interface")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)