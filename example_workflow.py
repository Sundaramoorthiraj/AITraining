#!/usr/bin/env python3
"""
Example workflow demonstrating the complete Resume Parser LLM training pipeline
"""

import os
import json
from pathlib import Path
import subprocess
import sys

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

def create_sample_resumes():
    """Create sample resume files for demonstration"""
    sample_dir = Path("sample_resumes")
    sample_dir.mkdir(exist_ok=True)
    
    # Sample resume 1
    resume1 = """John Smith
Software Engineer
Email: john.smith@email.com
Phone: (555) 987-6543
LinkedIn: https://linkedin.com/in/johnsmith
GitHub: https://github.com/johnsmith

EDUCATION
Bachelor of Science in Computer Science
University of California, Berkeley, 2019
GPA: 3.8/4.0

Master of Science in Software Engineering
Stanford University, 2021
GPA: 3.9/4.0

PROFESSIONAL EXPERIENCE
Senior Software Engineer at Google (2021-2024)
- Led development of microservices architecture
- Improved system performance by 50%
- Mentored junior developers
- Technologies: Python, Java, Kubernetes, Docker

Software Engineer at Microsoft (2019-2021)
- Developed web applications using React and Node.js
- Collaborated with cross-functional teams
- Implemented CI/CD pipelines
- Technologies: JavaScript, TypeScript, Azure

PROJECTS
E-commerce Platform
- Built full-stack application with Django and PostgreSQL
- Implemented payment processing with Stripe
- Deployed on AWS with Docker containers
- Technologies: Python, Django, PostgreSQL, AWS

Task Management App
- React Native mobile application
- Real-time collaboration features
- Used Firebase for backend services
- Technologies: React Native, Firebase, JavaScript

Machine Learning Model
- Developed recommendation system using TensorFlow
- Achieved 85% accuracy on test dataset
- Deployed model using TensorFlow Serving
- Technologies: Python, TensorFlow, scikit-learn

SKILLS
Programming Languages: Python, Java, JavaScript, TypeScript, SQL
Frameworks: React, Node.js, Django, Flask, Spring Boot
Cloud Platforms: AWS, Azure, Google Cloud
Tools: Docker, Kubernetes, Git, Jenkins, TensorFlow, PyTorch
"""
    
    with open(sample_dir / "resume1.txt", "w") as f:
        f.write(resume1)
    
    # Sample resume 2
    resume2 = """Sarah Johnson
Data Scientist
sarah.johnson@techcorp.com
+1 (555) 123-4567
https://linkedin.com/in/sarahjohnson
https://github.com/sarahjohnson

EDUCATION
Ph.D. in Machine Learning
Massachusetts Institute of Technology, 2020
Thesis: "Deep Learning for Natural Language Processing"

Bachelor of Science in Mathematics
Harvard University, 2016
Magna Cum Laude

EXPERIENCE
Senior Data Scientist at TechCorp (2020-Present)
- Developed ML models for customer segmentation
- Led team of 5 data scientists
- Improved model accuracy by 30%
- Technologies: Python, R, TensorFlow, PyTorch, Spark

Data Scientist at StartupXYZ (2018-2020)
- Built recommendation systems
- Analyzed user behavior data
- Created data visualization dashboards
- Technologies: Python, SQL, Tableau, scikit-learn

PROJECTS
Fraud Detection System
- Built real-time fraud detection using deep learning
- Reduced false positives by 25%
- Deployed on AWS with real-time processing
- Technologies: Python, TensorFlow, AWS, Kafka

Customer Analytics Dashboard
- Created interactive dashboard for business insights
- Integrated multiple data sources
- Used for executive decision making
- Technologies: Python, Dash, PostgreSQL, Redis

SKILLS
Machine Learning: Deep Learning, NLP, Computer Vision, Time Series
Programming: Python, R, SQL, Scala
Tools: TensorFlow, PyTorch, Spark, Hadoop, AWS, Docker
Statistics: Statistical Modeling, A/B Testing, Experimental Design
"""
    
    with open(sample_dir / "resume2.txt", "w") as f:
        f.write(resume2)
    
    print(f"✅ Created {len(list(sample_dir.glob('*.txt')))} sample resume files")

def demonstrate_workflow():
    """Demonstrate the complete workflow"""
    print("🚀 Resume Parser LLM Training - Complete Workflow Demo")
    print("=" * 60)
    
    # Step 1: Create sample data
    print("\n📊 Step 1: Creating sample resume data...")
    create_sample_resumes()
    
    # Step 2: Prepare dataset
    print("\n📊 Step 2: Preparing training dataset...")
    if not run_command(
        "python dataset_preparation.py --input_dir ./sample_resumes --output_dir ./dataset",
        "Dataset preparation"
    ):
        print("❌ Dataset preparation failed. Please check the script.")
        return False
    
    # Step 3: Check if we have training data
    dataset_path = Path("dataset/training_data")
    if not (dataset_path / "train.jsonl").exists():
        print("❌ Training data not found. Please check dataset preparation.")
        return False
    
    # Count training examples
    with open(dataset_path / "train.jsonl", "r") as f:
        train_count = sum(1 for line in f)
    
    with open(dataset_path / "validation.jsonl", "r") as f:
        val_count = sum(1 for line in f)
    
    print(f"📈 Dataset prepared: {train_count} training, {val_count} validation examples")
    
    # Step 4: Show training configuration
    print("\n⚙️ Step 3: Training configuration...")
    print("For actual training, you would run:")
    print("python train_resume_parser.py \\")
    print("    --model_name meta-llama/Llama-3.1-8B-Instruct \\")
    print("    --dataset_path ./dataset \\")
    print("    --output_dir ./models/resume_parser_llama3 \\")
    print("    --epochs 3 \\")
    print("    --batch_size 4")
    
    # Step 5: Show evaluation
    print("\n📈 Step 4: Model evaluation...")
    print("After training, you would run:")
    print("python evaluate_model.py \\")
    print("    --model_path ./models/resume_parser_llama3 \\")
    print("    --test_dataset_path ./dataset \\")
    print("    --output_path ./evaluations/results.json")
    
    # Step 6: Show inference
    print("\n🔮 Step 5: Model inference...")
    print("For parsing new resumes:")
    print("python inference.py \\")
    print("    --model_path ./models/resume_parser_llama3 \\")
    print("    --input resume.pdf \\")
    print("    --output parsed_result.json")
    
    # Step 7: Show sample training data
    print("\n📋 Step 6: Sample training data structure...")
    with open(dataset_path / "train.jsonl", "r") as f:
        sample_data = json.loads(f.readline())
    
    print("Training example structure:")
    print(f"  - Prompt length: {len(sample_data['prompt'])} characters")
    print(f"  - Response length: {len(sample_data['response'])} characters")
    print(f"  - File: {sample_data['file_path']}")
    
    # Show a snippet of the prompt
    prompt_snippet = sample_data['prompt'][:200] + "..."
    print(f"\nPrompt snippet:\n{prompt_snippet}")
    
    # Show a snippet of the response
    response_snippet = sample_data['response'][:200] + "..."
    print(f"\nResponse snippet:\n{response_snippet}")
    
    print("\n" + "=" * 60)
    print("✅ Workflow demonstration completed!")
    print("\n📚 Next steps:")
    print("1. Add more resume files to ./sample_resumes/")
    print("2. Run the training script with your preferred model")
    print("3. Evaluate the trained model")
    print("4. Use the model for inference")
    print("\n🔧 For production use:")
    print("- Use larger datasets (1000+ resumes)")
    print("- Train for more epochs")
    print("- Use larger models for better performance")
    print("- Implement proper validation and testing")
    
    return True

def main():
    """Main function"""
    try:
        success = demonstrate_workflow()
        if success:
            print("\n🎉 Demo completed successfully!")
        else:
            print("\n❌ Demo failed. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()