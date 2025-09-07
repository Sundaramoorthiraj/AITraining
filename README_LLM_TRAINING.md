# Resume Parser LLM Training

A comprehensive solution for fine-tuning Large Language Models (LLMs) like Llama 3 for resume parsing. This approach replaces regex-based pattern matching with intelligent LLM-based extraction that can handle diverse resume formats and structures.

## 🎯 Overview

This project provides a complete pipeline for:
- **Dataset Preparation**: Converting resume files into training data
- **Model Training**: Fine-tuning LLMs for resume parsing
- **Evaluation**: Testing model performance
- **Inference**: Using trained models for resume parsing

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or download the project files
# Install dependencies and setup environment
python setup_training.py

# Or manually install dependencies
pip install -r requirements_training.txt
```

### 2. Prepare Your Dataset

```bash
# Place your resume files in a directory
mkdir sample_resumes
# Copy your resume files (PDF, DOC, DOCX, RTF, TXT, images) to sample_resumes/

# Generate training dataset
python dataset_preparation.py --input_dir ./sample_resumes --output_dir ./dataset
```

### 3. Train the Model

```bash
# Train with Llama 3.1 8B
python train_resume_parser.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path ./dataset \
    --output_dir ./models/resume_parser_llama3 \
    --epochs 3 \
    --batch_size 4
```

### 4. Evaluate the Model

```bash
# Evaluate trained model
python evaluate_model.py \
    --model_path ./models/resume_parser_llama3 \
    --test_dataset_path ./dataset \
    --output_path ./evaluations/results.json
```

### 5. Use for Inference

```bash
# Parse a resume using trained model
python inference.py \
    --model_path ./models/resume_parser_llama3 \
    --input resume.pdf \
    --output parsed_result.json
```

## 📁 Project Structure

```
resume-parser-llm/
├── dataset_preparation.py      # Dataset creation pipeline
├── train_resume_parser.py      # Model training script
├── evaluate_model.py           # Model evaluation
├── inference.py                # Inference script
├── setup_training.py           # Environment setup
├── requirements_training.txt   # Training dependencies
├── training_config.json        # Training configuration
├── dataset_config.json         # Dataset configuration
├── quick_start.sh              # Quick start script
├── batch_training.sh           # Batch training script
├── dataset/                    # Generated training data
│   ├── raw_data/              # Raw parsed resumes
│   ├── processed_data/        # Processed data
│   └── training_data/         # Training/validation splits
├── models/                     # Trained models
├── evaluations/                # Evaluation results
└── logs/                       # Training logs
```

## 🔧 Detailed Usage

### Dataset Preparation

The `dataset_preparation.py` script converts resume files into training data:

```bash
python dataset_preparation.py \
    --input_dir ./sample_resumes \
    --output_dir ./dataset \
    --train_ratio 0.8
```

**Parameters:**
- `--input_dir`: Directory containing resume files
- `--output_dir`: Output directory for processed data
- `--train_ratio`: Ratio of data for training (default: 0.8)

**Supported Formats:**
- PDF (.pdf)
- Word Documents (.doc, .docx)
- Rich Text Format (.rtf)
- Text Files (.txt)
- Images (.jpg, .jpeg, .png)

**Output Structure:**
```
dataset/
├── raw_data/           # Individual parsed resumes
├── processed_data/     # Processed training examples
└── training_data/      # Train/validation splits
    ├── train.jsonl     # Training data
    ├── validation.jsonl # Validation data
    ├── train.csv       # CSV format
    └── validation.csv  # CSV format
```

### Model Training

The `train_resume_parser.py` script fine-tunes LLMs:

```bash
python train_resume_parser.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path ./dataset \
    --output_dir ./models/resume_parser_llama3 \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4
```

**Parameters:**
- `--model_name`: Hugging Face model name
- `--dataset_path`: Path to training dataset
- `--output_dir`: Output directory for trained model
- `--epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--learning_rate`: Learning rate

**Supported Models:**
- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.1-70B-Instruct`
- `microsoft/DialoGPT-medium`
- Any Hugging Face causal LM model

**Training Features:**
- 4-bit quantization for memory efficiency
- Gradient accumulation
- Automatic mixed precision (AMP)
- Model checkpointing
- Validation monitoring

### Model Evaluation

The `evaluate_model.py` script tests model performance:

```bash
python evaluate_model.py \
    --model_path ./models/resume_parser_llama3 \
    --test_dataset_path ./dataset \
    --output_path ./evaluations/results.json
```

**Metrics Calculated:**
- Overall accuracy
- Field-wise accuracy (name, email, phone, etc.)
- Precision, recall, F1-score
- Detailed per-resume results

### Inference

The `inference.py` script uses trained models for parsing:

```bash
# Parse from file
python inference.py \
    --model_path ./models/resume_parser_llama3 \
    --input resume.pdf \
    --output parsed_result.json

# Parse from text
python inference.py \
    --model_path ./models/resume_parser_llama3 \
    --input "John Doe Software Engineer..." \
    --text \
    --output parsed_result.json
```

## ⚙️ Configuration

### Training Configuration

Edit `training_config.json` to customize training:

```json
{
  "model_name": "meta-llama/Llama-3.1-8B-Instruct",
  "dataset_path": "./dataset",
  "output_dir": "./models/resume_parser_llama3",
  "training": {
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "gradient_accumulation_steps": 4,
    "max_length": 2048
  },
  "quantization": {
    "use_4bit": true,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "float16"
  }
}
```

### Dataset Configuration

Edit `dataset_config.json` to customize data processing:

```json
{
  "input_directory": "./sample_resumes",
  "output_directory": "./dataset",
  "supported_formats": [".pdf", ".doc", ".docx", ".rtf", ".txt", ".jpg", ".jpeg", ".png"],
  "max_file_size_mb": 16,
  "train_validation_split": 0.8
}
```

## 🎯 Training Data Format

The training data follows this structure:

**Input Prompt:**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert resume parser. Extract the following information from the given resume text and return it in the specified JSON format.

Required fields:
- first_name: Person's first name
- last_name: Person's last name  
- email: Email address(es) as list
- phone: Phone number(s) as list
- linkedin: LinkedIn profile URL(s) as list
- github: GitHub profile URL(s) as list
- education: List of education entries
- experience: List of work experience entries
- projects: List of projects
- skills: List of technical and soft skills

<|eot_id|><|start_header_id|>user<|end_header_id|>

Please parse the following resume text and extract the information in JSON format:

[RESUME TEXT HERE]

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

**Expected Output:**
```json
{
  "first_name": "John",
  "last_name": "Doe",
  "email": ["john.doe@email.com"],
  "phone": ["+1 (555) 123-4567"],
  "linkedin": ["https://linkedin.com/in/johndoe"],
  "github": ["https://github.com/johndoe"],
  "education": [
    "Bachelor of Science in Computer Science, University of Technology, 2020"
  ],
  "experience": [
    "Software Engineer at Tech Corp (2020-2023) - Developed web applications"
  ],
  "projects": [
    "E-commerce Platform - Built with Django and PostgreSQL"
  ],
  "skills": ["Python", "JavaScript", "React", "Django"]
}
```

## 🚀 Advanced Usage

### Batch Training Multiple Models

```bash
# Train multiple models with different configurations
./batch_training.sh
```

### Custom Model Integration

```python
from inference import ResumeParserInference

# Load your trained model
parser = ResumeParserInference("./models/resume_parser_llama3")

# Parse resume
result = parser.parse_resume("John Doe Software Engineer...")
print(result["parsed_data"])
```

### Web API Integration

```python
from flask import Flask, request, jsonify
from inference import ResumeParserInference

app = Flask(__name__)
parser = ResumeParserInference("./models/resume_parser_llama3")

@app.route('/parse', methods=['POST'])
def parse_resume():
    resume_text = request.json.get('text')
    result = parser.parse_resume(resume_text)
    return jsonify(result["parsed_data"])

if __name__ == '__main__':
    app.run(debug=True)
```

## 📊 Performance Optimization

### Memory Optimization
- Use 4-bit quantization for large models
- Adjust batch size based on GPU memory
- Use gradient accumulation for effective larger batch sizes

### Training Speed
- Use multiple GPUs with `accelerate`
- Enable mixed precision training
- Use efficient attention mechanisms

### Model Selection
- **Llama 3.1 8B**: Good balance of performance and resource usage
- **Llama 3.1 70B**: Best performance, requires significant resources
- **Smaller models**: Faster inference, lower accuracy

## 🔍 Troubleshooting

### Common Issues

1. **Out of Memory Error**
   ```bash
   # Reduce batch size
   python train_resume_parser.py --batch_size 2
   
   # Use gradient accumulation
   python train_resume_parser.py --gradient_accumulation_steps 8
   ```

2. **Poor Parsing Quality**
   - Increase training epochs
   - Add more diverse training data
   - Use larger model
   - Adjust learning rate

3. **Slow Training**
   - Use multiple GPUs
   - Enable mixed precision
   - Use smaller model for initial experiments

### Debug Mode

```bash
# Enable debug logging
export TRANSFORMERS_VERBOSITY=debug
python train_resume_parser.py --debug
```

## 📈 Evaluation Metrics

The evaluation script provides comprehensive metrics:

- **Overall Accuracy**: Percentage of correctly parsed fields
- **Field-wise Accuracy**: Accuracy for each field type
- **Precision/Recall/F1**: For classification tasks
- **Per-resume Analysis**: Detailed results for each test case

## 🎯 Best Practices

### Dataset Quality
- Use diverse resume formats and styles
- Include resumes from different industries
- Ensure high-quality annotations
- Balance training/validation splits

### Training Strategy
- Start with smaller models for experimentation
- Use validation set for hyperparameter tuning
- Monitor training metrics closely
- Save checkpoints regularly

### Model Deployment
- Test on unseen data before deployment
- Implement fallback mechanisms
- Monitor model performance in production
- Regular retraining with new data

## 📚 Additional Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Llama 3 Model Card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
- [Quantization Guide](https://huggingface.co/docs/transformers/quantization)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the configuration files
3. Ensure all dependencies are installed
4. Check GPU memory and compute requirements

---

**Happy Training! 🚀**