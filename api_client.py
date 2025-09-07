#!/usr/bin/env python3
"""
API Client for Resume Parser
Example usage of the Resume Parser API
"""

import requests
import json
import sys
import os

def parse_resume_api(file_path, api_url="http://localhost:5000/api/parse"):
    """
    Parse a resume using the API
    
    Args:
        file_path (str): Path to the resume file
        api_url (str): API endpoint URL
    
    Returns:
        dict: Parsed resume data or error information
    """
    
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    
    try:
        with open(file_path, 'rb') as file:
            files = {'file': file}
            response = requests.post(api_url, files=files)
            
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API request failed with status {response.status_code}: {response.text}"}
            
    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to API. Make sure the server is running on localhost:5000"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def print_results(data):
    """Print parsed results in a formatted way"""
    
    if "error" in data:
        print(f"❌ Error: {data['error']}")
        return
    
    print("✅ Resume parsed successfully!")
    print("=" * 60)
    
    # Personal Information
    if data.get('names'):
        names = data['names']
        if names.get('first_name') or names.get('last_name'):
            print(f"👤 Name: {names.get('first_name', '')} {names.get('last_name', '')}")
    
    # Contact Information
    if data.get('emails'):
        print(f"📧 Email: {', '.join(data['emails'])}")
    
    if data.get('phones'):
        print(f"📞 Phone: {', '.join(data['phones'])}")
    
    # Professional Links
    if data.get('linkedin'):
        print(f"💼 LinkedIn: {data['linkedin'][0]}")
    
    if data.get('github'):
        print(f"🐙 GitHub: {data['github'][0]}")
    
    # Education
    if data.get('education'):
        print(f"\n🎓 Education ({len(data['education'])} items):")
        for i, item in enumerate(data['education'][:3], 1):  # Show first 3
            print(f"   {i}. {item}")
        if len(data['education']) > 3:
            print(f"   ... and {len(data['education']) - 3} more")
    
    # Experience
    if data.get('experience'):
        print(f"\n💼 Experience ({len(data['experience'])} items):")
        for i, item in enumerate(data['experience'][:3], 1):  # Show first 3
            print(f"   {i}. {item}")
        if len(data['experience']) > 3:
            print(f"   ... and {len(data['experience']) - 3} more")
    
    # Projects
    if data.get('projects'):
        print(f"\n🚀 Projects ({len(data['projects'])} items):")
        for i, item in enumerate(data['projects'][:3], 1):  # Show first 3
            print(f"   {i}. {item}")
        if len(data['projects']) > 3:
            print(f"   ... and {len(data['projects']) - 3} more")
    
    # File Information
    print(f"\n📁 File: {data.get('file_name', 'Unknown')}")
    print(f"📄 Type: {data.get('file_type', 'Unknown')}")

def main():
    """Main function for command-line usage"""
    
    if len(sys.argv) != 2:
        print("Usage: python api_client.py <resume_file_path>")
        print("\nExample:")
        print("  python api_client.py sample_resume.pdf")
        print("  python api_client.py resume.docx")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    print(f"🔄 Parsing resume: {file_path}")
    print("=" * 60)
    
    # Parse the resume
    result = parse_resume_api(file_path)
    
    # Print results
    print_results(result)
    
    # Optionally save to JSON file
    if "error" not in result:
        output_file = f"parsed_{os.path.splitext(os.path.basename(file_path))[0]}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main()