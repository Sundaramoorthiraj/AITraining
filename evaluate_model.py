#!/usr/bin/env python3
"""
Evaluation script for the fine-tuned Resume Parser model
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from resume_parser import ResumeParser
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re

class ResumeParserEvaluator:
    def __init__(self, model_path: str, test_dataset_path: str):
        self.model_path = Path(model_path)
        self.test_dataset_path = Path(test_dataset_path)
        
        # Load model and tokenizer
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
        
        # Baseline parser for comparison
        self.baseline_parser = ResumeParser()
        
    def load_test_data(self) -> List[Dict[str, Any]]:
        """Load test dataset"""
        test_files = list(self.test_dataset_path.glob("**/*.json"))
        test_data = []
        
        for file_path in test_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                test_data.append(data)
        
        return test_data
    
    def generate_response(self, prompt: str, max_length: int = 1024) -> str:
        """Generate response using the fine-tuned model"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
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
        except:
            pass
        
        return {}
    
    def evaluate_single_resume(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single resume"""
        # Create prompt
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

{resume_data.get('raw_text', '')[:4000]}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Extract JSON
        predicted_data = self.extract_json_from_response(response)
        
        # Get ground truth
        ground_truth = {
            "first_name": resume_data.get('names', {}).get('first_name', ''),
            "last_name": resume_data.get('names', {}).get('last_name', ''),
            "email": resume_data.get('emails', []),
            "phone": resume_data.get('phones', []),
            "linkedin": resume_data.get('linkedin', []),
            "github": resume_data.get('github', []),
            "education": resume_data.get('education', []),
            "experience": resume_data.get('experience', []),
            "projects": resume_data.get('projects', [])
        }
        
        return {
            "predicted": predicted_data,
            "ground_truth": ground_truth,
            "file_name": resume_data.get('file_name', 'unknown')
        }
    
    def calculate_field_accuracy(self, predicted: Any, ground_truth: Any) -> float:
        """Calculate accuracy for a specific field"""
        if isinstance(ground_truth, list) and isinstance(predicted, list):
            # For lists, check if any items match
            if not ground_truth and not predicted:
                return 1.0
            if not ground_truth or not predicted:
                return 0.0
            
            # Check for any overlap
            ground_truth_set = set(str(item).lower() for item in ground_truth)
            predicted_set = set(str(item).lower() for item in predicted)
            
            if ground_truth_set & predicted_set:  # Intersection
                return 1.0
            else:
                return 0.0
        
        elif isinstance(ground_truth, str) and isinstance(predicted, str):
            return 1.0 if ground_truth.lower() == predicted.lower() else 0.0
        
        return 0.0
    
    def evaluate_all(self) -> Dict[str, Any]:
        """Evaluate all test resumes"""
        test_data = self.load_test_data()
        results = []
        
        print(f"Evaluating {len(test_data)} resumes...")
        
        for i, resume_data in enumerate(test_data):
            print(f"Processing {i+1}/{len(test_data)}: {resume_data.get('file_name', 'unknown')}")
            
            result = self.evaluate_single_resume(resume_data)
            results.append(result)
        
        # Calculate overall metrics
        field_accuracies = {
            "first_name": [],
            "last_name": [],
            "email": [],
            "phone": [],
            "linkedin": [],
            "github": [],
            "education": [],
            "experience": [],
            "projects": []
        }
        
        for result in results:
            predicted = result["predicted"]
            ground_truth = result["ground_truth"]
            
            for field in field_accuracies.keys():
                accuracy = self.calculate_field_accuracy(
                    predicted.get(field, ""),
                    ground_truth.get(field, "")
                )
                field_accuracies[field].append(accuracy)
        
        # Calculate average accuracies
        avg_accuracies = {}
        for field, accuracies in field_accuracies.items():
            avg_accuracies[field] = sum(accuracies) / len(accuracies) if accuracies else 0.0
        
        # Overall accuracy
        overall_accuracy = sum(avg_accuracies.values()) / len(avg_accuracies)
        
        evaluation_results = {
            "overall_accuracy": overall_accuracy,
            "field_accuracies": avg_accuracies,
            "total_resumes": len(test_data),
            "detailed_results": results
        }
        
        return evaluation_results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary report
        summary_path = output_path.replace('.json', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Resume Parser Model Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Overall Accuracy: {results['overall_accuracy']:.3f}\n\n")
            f.write("Field-wise Accuracies:\n")
            for field, accuracy in results['field_accuracies'].items():
                f.write(f"  {field}: {accuracy:.3f}\n")
            f.write(f"\nTotal Resumes Evaluated: {results['total_resumes']}\n")
        
        print(f"Results saved to {output_path}")
        print(f"Summary saved to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Resume Parser Model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the fine-tuned model")
    parser.add_argument("--test_dataset_path", type=str, required=True,
                       help="Path to test dataset")
    parser.add_argument("--output_path", type=str, default="evaluation_results.json",
                       help="Output path for evaluation results")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ResumeParserEvaluator(args.model_path, args.test_dataset_path)
    
    # Run evaluation
    results = evaluator.evaluate_all()
    
    # Save results
    evaluator.save_results(results, args.output_path)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Overall Accuracy: {results['overall_accuracy']:.3f}")
    print("\nField-wise Accuracies:")
    for field, accuracy in results['field_accuracies'].items():
        print(f"  {field}: {accuracy:.3f}")

if __name__ == "__main__":
    main()