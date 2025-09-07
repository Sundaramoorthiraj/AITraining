#!/usr/bin/env python3
"""
Setup script for Resume Parser LLM Training Environment
"""

import subprocess
import sys
import os
import json
from pathlib import Path

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

def create_directories():
    """Create necessary directories"""
    directories = [
        "dataset/raw_data",
        "dataset/processed_data", 
        "dataset/training_data",
        "models",
        "logs",
        "evaluations"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def create_sample_configs():
    """Create sample configuration files"""
    
    # Training configuration
    training_config = {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "dataset_path": "./dataset",
        "output_dir": "./models/resume_parser_llama3",
        "training": {
            "num_epochs": 3,
            "batch_size": 4,
            "learning_rate": 2e-4,
            "warmup_steps": 100,
            "gradient_accumulation_steps": 4,
            "max_length": 2048,
            "save_steps": 500,
            "eval_steps": 500
        },
        "quantization": {
            "use_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "float16"
        },
        "data": {
            "train_ratio": 0.8,
            "max_resumes": 1000
        }
    }
    
    with open("training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)
    
    # Dataset configuration
    dataset_config = {
        "input_directory": "./sample_resumes",
        "output_directory": "./dataset",
        "supported_formats": [".pdf", ".doc", ".docx", ".rtf", ".txt", ".jpg", ".jpeg", ".png"],
        "max_file_size_mb": 16,
        "train_validation_split": 0.8
    }
    
    with open("dataset_config.json", "w") as f:
        json.dump(dataset_config, f, indent=2)
    
    print("📋 Created configuration files")

def create_sample_scripts():
    """Create sample training and evaluation scripts"""
    
    # Quick start script
    quick_start = """#!/bin/bash
# Quick Start Script for Resume Parser Training

echo "🚀 Resume Parser LLM Training - Quick Start"
echo "=========================================="

# Step 1: Prepare dataset
echo "📊 Step 1: Preparing dataset..."
python dataset_preparation.py --input_dir ./sample_resumes --output_dir ./dataset

# Step 2: Train model
echo "🤖 Step 2: Training model..."
python train_resume_parser.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset_path ./dataset --output_dir ./models/resume_parser_llama3

# Step 3: Evaluate model
echo "📈 Step 3: Evaluating model..."
python evaluate_model.py --model_path ./models/resume_parser_llama3 --test_dataset_path ./dataset --output_path ./evaluations/results.json

echo "✅ Training pipeline completed!"
"""
    
    with open("quick_start.sh", "w") as f:
        f.write(quick_start)
    
    os.chmod("quick_start.sh", 0o755)
    
    # Batch training script
    batch_script = """#!/bin/bash
# Batch Training Script for Multiple Models

models=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.1-70B-Instruct"
    "microsoft/DialoGPT-medium"
)

for model in "${models[@]}"; do
    echo "Training with model: $model"
    model_name=$(basename "$model")
    python train_resume_parser.py \\
        --model_name "$model" \\
        --dataset_path ./dataset \\
        --output_dir "./models/resume_parser_$model_name" \\
        --epochs 3 \\
        --batch_size 4
done
"""
    
    with open("batch_training.sh", "w") as f:
        f.write(batch_script)
    
    os.chmod("batch_training.sh", 0o755)
    
    print("📜 Created sample scripts")

def main():
    print("🚀 Setting up Resume Parser LLM Training Environment...")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Install Python dependencies
    if not run_command("pip install -r requirements_training.txt", "Installing training dependencies"):
        print("❌ Failed to install dependencies. Please check your Python environment.")
        return False
    
    # Install system dependencies
    if not run_command("sudo apt update && sudo apt install -y libmagic1 tesseract-ocr", "Installing system dependencies"):
        print("⚠️  System dependencies installation failed, but continuing...")
    
    # Create configuration files
    create_sample_configs()
    
    # Create sample scripts
    create_sample_scripts()
    
    print("\n" + "=" * 60)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Place your resume files in ./sample_resumes/ directory")
    print("2. Run: python dataset_preparation.py --input_dir ./sample_resumes --output_dir ./dataset")
    print("3. Run: python train_resume_parser.py --model_name meta-llama/Llama-3.1-8B-Instruct")
    print("4. Run: python evaluate_model.py --model_path ./models/resume_parser_llama3")
    print("\n🔧 Quick start:")
    print("   ./quick_start.sh")
    print("\n📚 Configuration files created:")
    print("   - training_config.json")
    print("   - dataset_config.json")
    print("\n🎯 For production use:")
    print("   - Adjust training parameters in training_config.json")
    print("   - Use larger models for better performance")
    print("   - Increase dataset size for better generalization")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)