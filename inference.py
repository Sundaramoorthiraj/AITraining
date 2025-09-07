#!/usr/bin/env python3
"""
Inference script for the fine-tuned Resume Parser model
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

class ResumeParserInference:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        
        # Load model and tokenizer
        print(f"Loading model from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        print("Model loaded successfully!")
    
    def create_prompt(self, resume_text: str) -> str:
        """Create the prompt for resume parsing"""
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert resume parser. Extract the following information from the given resume text and return it in the specified JSON format.

Required fields:
- first_name: Person's first name
- last_name: Person's last name  
- email: Email address(es) as list
- phone: Phone number(s) as list
- linkedin: LinkedIn profile URL(s) as list
- github: GitHub profile URL(s) as list
- education: List of education entries (degree, institution, year, GPA if available)
- experience: List of work experience entries (job title, company, duration, key responsibilities)
- projects: List of projects (project name, description, technologies used)
- skills: List of technical and soft skills

<|eot_id|><|start_header_id|>user<|end_header_id|>

Please parse the following resume text and extract the information in JSON format:

{resume_text}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
    
    def generate_response(self, prompt: str, max_length: int = 1024) -> str:
        """Generate response using the fine-tuned model"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from model response"""
        try:
            # Find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
        except Exception as e:
            print(f"Error parsing JSON: {e}")
        
        return {}
    
    def parse_resume(self, resume_text: str) -> Dict[str, Any]:
        """Parse a resume and return structured data"""
        # Truncate if too long
        if len(resume_text) > 4000:
            resume_text = resume_text[:4000] + "..."
        
        # Create prompt
        prompt = self.create_prompt(resume_text)
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Extract JSON
        parsed_data = self.extract_json_from_response(response)
        
        return {
            "parsed_data": parsed_data,
            "raw_response": response
        }
    
    def parse_resume_file(self, file_path: str) -> Dict[str, Any]:
        """Parse a resume file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file content
        if file_path.suffix.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                resume_text = f.read()
        else:
            # For other formats, you might want to use the original parser
            # to extract text first, then use the LLM
            from resume_parser import ResumeParser
            parser = ResumeParser()
            result = parser.parse_resume(str(file_path))
            resume_text = result.get('raw_text', '')
        
        return self.parse_resume(resume_text)

def main():
    parser = argparse.ArgumentParser(description="Resume Parser Inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the fine-tuned model")
    parser.add_argument("--input", type=str, required=True,
                       help="Input resume file or text")
    parser.add_argument("--output", type=str, default="parsed_resume.json",
                       help="Output file for parsed results")
    parser.add_argument("--text", action="store_true",
                       help="Treat input as raw text instead of file path")
    
    args = parser.parse_args()
    
    # Create inference engine
    inference_engine = ResumeParserInference(args.model_path)
    
    # Parse resume
    if args.text:
        # Input is raw text
        result = inference_engine.parse_resume(args.input)
    else:
        # Input is file path
        result = inference_engine.parse_resume_file(args.input)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Resume parsed successfully!")
    print(f"Results saved to: {args.output}")
    
    # Print summary
    parsed_data = result["parsed_data"]
    print("\nParsed Information:")
    print(f"Name: {parsed_data.get('first_name', '')} {parsed_data.get('last_name', '')}")
    print(f"Email: {parsed_data.get('email', [])}")
    print(f"Phone: {parsed_data.get('phone', [])}")
    print(f"LinkedIn: {parsed_data.get('linkedin', [])}")
    print(f"GitHub: {parsed_data.get('github', [])}")
    print(f"Education entries: {len(parsed_data.get('education', []))}")
    print(f"Experience entries: {len(parsed_data.get('experience', []))}")
    print(f"Projects: {len(parsed_data.get('projects', []))}")
    print(f"Skills: {len(parsed_data.get('skills', []))}")

if __name__ == "__main__":
    main()