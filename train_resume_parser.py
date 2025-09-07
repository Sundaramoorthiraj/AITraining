#!/usr/bin/env python3
"""
Training script for Resume Parser LLM Fine-tuning
Supports multiple training frameworks: Transformers, Unsloth, Axolotl
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from datasets import Dataset
import pandas as pd

class ResumeParserTrainer:
    def __init__(self, model_name: str, dataset_path: str, output_dir: str):
        self.model_name = model_name
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load tokenizer and model
        self.tokenizer = None
        self.model = None
        
    def load_model_and_tokenizer(self, use_quantization: bool = True):
        """Load the model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization for memory efficiency
        if use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            quantization_config = None
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if use_quantization else torch.float32
        )
        
        print("Model and tokenizer loaded successfully!")
    
    def load_dataset(self) -> Dataset:
        """Load and prepare the training dataset"""
        print("Loading dataset...")
        
        # Load training data
        train_file = self.dataset_path / "training_data" / "train.jsonl"
        val_file = self.dataset_path / "training_data" / "validation.jsonl"
        
        if not train_file.exists():
            raise FileNotFoundError(f"Training file not found: {train_file}")
        
        # Load data
        train_data = []
        with open(train_file, 'r') as f:
            for line in f:
                train_data.append(json.loads(line.strip()))
        
        val_data = []
        if val_file.exists():
            with open(val_file, 'r') as f:
                for line in f:
                    val_data.append(json.loads(line.strip()))
        
        print(f"Loaded {len(train_data)} training examples")
        print(f"Loaded {len(val_data)} validation examples")
        
        # Create datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data) if val_data else None
        
        return train_dataset, val_dataset
    
    def tokenize_function(self, examples):
        """Tokenize the examples"""
        # Combine prompt and response
        texts = []
        for i in range(len(examples['prompt'])):
            text = examples['prompt'][i] + examples['response'][i]
            texts.append(text)
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=2048,
            return_tensors="pt"
        )
        
        # Set labels (same as input_ids for causal LM)
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    def prepare_datasets(self, train_dataset, val_dataset):
        """Prepare datasets for training"""
        print("Tokenizing datasets...")
        
        # Tokenize training data
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        # Tokenize validation data if exists
        if val_dataset:
            val_dataset = val_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=val_dataset.column_names
            )
        
        return train_dataset, val_dataset
    
    def train(self, 
              num_epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 2e-4,
              warmup_steps: int = 100,
              save_steps: int = 500,
              eval_steps: int = 500,
              gradient_accumulation_steps: int = 4):
        """Train the model"""
        
        print("Starting training...")
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Load and prepare datasets
        train_dataset, val_dataset = self.load_dataset()
        train_dataset, val_dataset = self.prepare_datasets(train_dataset, val_dataset)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=50,
            save_steps=save_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps" if val_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            greater_is_better=False,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=None,  # Disable wandb/tensorboard
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"Training completed! Model saved to: {self.output_dir}")
        
        return trainer

def create_training_config():
    """Create a training configuration file"""
    config = {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "dataset_path": "./dataset",
        "output_dir": "./resume_parser_model",
        "training": {
            "num_epochs": 3,
            "batch_size": 4,
            "learning_rate": 2e-4,
            "warmup_steps": 100,
            "gradient_accumulation_steps": 4,
            "max_length": 2048
        },
        "quantization": {
            "use_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "float16"
        }
    }
    
    with open("training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Training configuration saved to training_config.json")

def main():
    parser = argparse.ArgumentParser(description="Train Resume Parser LLM")
    parser.add_argument("--model_name", type=str, 
                       default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Model name to fine-tune")
    parser.add_argument("--dataset_path", type=str, default="./dataset",
                       help="Path to training dataset")
    parser.add_argument("--output_dir", type=str, default="./resume_parser_model",
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--create_config", action="store_true",
                       help="Create training configuration file")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_training_config()
        return
    
    # Check if dataset exists
    if not Path(args.dataset_path).exists():
        print(f"Dataset not found at {args.dataset_path}")
        print("Please run dataset_preparation.py first")
        return
    
    # Create trainer
    trainer = ResumeParserTrainer(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir
    )
    
    # Train the model
    trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

if __name__ == "__main__":
    main()