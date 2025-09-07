#!/usr/bin/env python3
"""
Dataset Preparation for Resume Parser LLM Fine-tuning
Creates training data from resume files and structured annotations
"""

import os
import json
import random
from typing import Dict, List, Any
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
from resume_parser import ResumeParser
import argparse

@dataclass
class ResumeData:
    """Data structure for resume information"""
    raw_text: str
    first_name: str
    last_name: str
    email: str
    phone: str
    linkedin: str
    github: str
    education: List[str]
    experience: List[str]
    projects: List[str]
    skills: List[str]
    file_path: str

class DatasetPreparator:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.parser = ResumeParser()
        
        # Create subdirectories
        (self.output_dir / "raw_data").mkdir(exist_ok=True)
        (self.output_dir / "processed_data").mkdir(exist_ok=True)
        (self.output_dir / "training_data").mkdir(exist_ok=True)
        
    def collect_resume_files(self) -> List[Path]:
        """Collect all resume files from input directory"""
        resume_files = []
        extensions = ['.pdf', '.doc', '.docx', '.rtf', '.txt', '.jpg', '.jpeg', '.png']
        
        for ext in extensions:
            resume_files.extend(self.input_dir.glob(f"**/*{ext}"))
            
        print(f"Found {len(resume_files)} resume files")
        return resume_files
    
    def parse_resume_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse a single resume file and extract structured data"""
        try:
            result = self.parser.parse_resume(str(file_path))
            return result
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return {"error": str(e)}
    
    def create_structured_prompt(self, resume_data: Dict[str, Any]) -> str:
        """Create a structured prompt for training"""
        prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

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

"""
        
        # Add the resume text
        resume_text = resume_data.get('raw_text', '')
        if len(resume_text) > 4000:  # Truncate very long resumes
            resume_text = resume_text[:4000] + "..."
            
        prompt += resume_text
        prompt += "\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return prompt
    
    def create_expected_response(self, resume_data: Dict[str, Any]) -> str:
        """Create the expected JSON response for training"""
        # Clean and structure the data
        response_data = {
            "first_name": resume_data.get('names', {}).get('first_name', ''),
            "last_name": resume_data.get('names', {}).get('last_name', ''),
            "email": resume_data.get('emails', []),
            "phone": resume_data.get('phones', []),
            "linkedin": resume_data.get('linkedin', []),
            "github": resume_data.get('github', []),
            "education": resume_data.get('education', []),
            "experience": resume_data.get('experience', []),
            "projects": resume_data.get('projects', []),
            "skills": self.extract_skills_from_text(resume_data.get('raw_text', ''))
        }
        
        return json.dumps(response_data, indent=2)
    
    def extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from resume text using keyword matching"""
        skills_keywords = [
            'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue',
            'node.js', 'django', 'flask', 'spring', 'sql', 'postgresql', 'mysql',
            'mongodb', 'redis', 'docker', 'kubernetes', 'aws', 'azure', 'gcp',
            'git', 'jenkins', 'ci/cd', 'agile', 'scrum', 'machine learning',
            'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy',
            'html', 'css', 'bootstrap', 'jquery', 'rest api', 'graphql',
            'microservices', 'api development', 'web development', 'mobile development'
        ]
        
        found_skills = []
        text_lower = text.lower()
        
        for skill in skills_keywords:
            if skill in text_lower:
                found_skills.append(skill.title())
        
        return list(set(found_skills))
    
    def create_training_example(self, resume_data: Dict[str, Any]) -> Dict[str, str]:
        """Create a single training example"""
        prompt = self.create_structured_prompt(resume_data)
        response = self.create_expected_response(resume_data)
        
        return {
            "prompt": prompt,
            "response": response,
            "file_path": resume_data.get('file_name', 'unknown')
        }
    
    def process_all_resumes(self) -> List[Dict[str, str]]:
        """Process all resume files and create training examples"""
        resume_files = self.collect_resume_files()
        training_examples = []
        
        for i, file_path in enumerate(resume_files):
            print(f"Processing {i+1}/{len(resume_files)}: {file_path.name}")
            
            # Parse the resume
            resume_data = self.parse_resume_file(file_path)
            
            if "error" in resume_data:
                print(f"Skipping {file_path.name} due to error")
                continue
            
            # Create training example
            example = self.create_training_example(resume_data)
            training_examples.append(example)
            
            # Save raw data
            raw_data_path = self.output_dir / "raw_data" / f"{file_path.stem}.json"
            with open(raw_data_path, 'w') as f:
                json.dump(resume_data, f, indent=2)
        
        return training_examples
    
    def save_training_data(self, training_examples: List[Dict[str, str]], 
                          train_ratio: float = 0.8):
        """Save training data in different formats"""
        
        # Shuffle the data
        random.shuffle(training_examples)
        
        # Split into train/validation
        split_idx = int(len(training_examples) * train_ratio)
        train_data = training_examples[:split_idx]
        val_data = training_examples[split_idx:]
        
        # Save as JSON
        with open(self.output_dir / "training_data" / "train.json", 'w') as f:
            json.dump(train_data, f, indent=2)
            
        with open(self.output_dir / "training_data" / "validation.json", 'w') as f:
            json.dump(val_data, f, indent=2)
        
        # Save as JSONL for some training frameworks
        with open(self.output_dir / "training_data" / "train.jsonl", 'w') as f:
            for example in train_data:
                f.write(json.dumps(example) + '\n')
                
        with open(self.output_dir / "training_data" / "validation.jsonl", 'w') as f:
            for example in val_data:
                f.write(json.dumps(example) + '\n')
        
        # Save as CSV for analysis
        df_train = pd.DataFrame(train_data)
        df_val = pd.DataFrame(val_data)
        
        df_train.to_csv(self.output_dir / "training_data" / "train.csv", index=False)
        df_val.to_csv(self.output_dir / "training_data" / "validation.csv", index=False)
        
        print(f"Training data saved:")
        print(f"  - Train examples: {len(train_data)}")
        print(f"  - Validation examples: {len(val_data)}")
        print(f"  - Total examples: {len(training_examples)}")
        
        return len(training_examples)

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for resume parser training")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing resume files")
    parser.add_argument("--output_dir", type=str, default="./dataset",
                       help="Output directory for processed data")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Ratio of data to use for training")
    
    args = parser.parse_args()
    
    # Create dataset preparator
    preparator = DatasetPreparator(args.input_dir, args.output_dir)
    
    # Process all resumes
    print("Starting dataset preparation...")
    training_examples = preparator.process_all_resumes()
    
    if not training_examples:
        print("No valid training examples created!")
        return
    
    # Save training data
    total_examples = preparator.save_training_data(training_examples, args.train_ratio)
    
    print(f"\nDataset preparation complete!")
    print(f"Total examples created: {total_examples}")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()