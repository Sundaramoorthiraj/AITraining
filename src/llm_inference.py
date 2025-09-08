"""
Local LLM inference module for CPU-based text generation
"""
import logging
import torch
from typing import List, Dict, Any, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)

from config import Config

logger = logging.getLogger(__name__)


class LocalLLMInference:
    """Handles local LLM inference for CPU-based text generation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = "cpu"  # Force CPU usage
        
    def load_model(self, model_name: str = None) -> None:
        """Load the LLM model and tokenizer"""
        if model_name is None:
            model_name = self.config.llm_model
        
        try:
            logger.info(f"Loading LLM model: {model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with CPU optimization
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,  # CPU
                torch_dtype=torch.float32
            )
            
            logger.info("LLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading LLM model: {e}")
            # Fallback to a smaller model
            self._load_fallback_model()
    
    def _load_fallback_model(self) -> None:
        """Load a fallback model if the primary model fails"""
        try:
            fallback_model = "microsoft/DialoGPT-small"
            logger.info(f"Loading fallback model: {fallback_model}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                fallback_model,
                cache_dir=self.config.cache_dir
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                fallback_model,
                cache_dir=self.config.cache_dir,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,
                torch_dtype=torch.float32
            )
            
            logger.info("Fallback model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading fallback model: {e}")
            raise
    
    def generate_response(self, prompt: str, max_tokens: int = None, 
                         temperature: float = None, top_p: float = None) -> str:
        """Generate a response using the loaded model"""
        if self.pipeline is None:
            self.load_model()
        
        if max_tokens is None:
            max_tokens = self.config.max_tokens
        if temperature is None:
            temperature = self.config.temperature
        if top_p is None:
            top_p = self.config.top_p
        
        try:
            # Generate response
            response = self.pipeline(
                prompt,
                max_length=len(self.tokenizer.encode(prompt)) + max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            
            # Remove the original prompt from the response
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating a response."
    
    def create_rag_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Create a prompt for RAG-based question answering"""
        if not context_chunks:
            return f"Question: {query}\nAnswer:"
        
        # Build context from retrieved chunks
        context = "Context:\n"
        for i, chunk_data in enumerate(context_chunks, 1):
            chunk = chunk_data['chunk']
            context += f"{i}. {chunk['text']}\n"
        
        # Create the prompt
        prompt = f"""{context}

Question: {query}

Based on the context provided above, please provide a comprehensive and accurate answer. If the context doesn't contain enough information to answer the question, please say so.

Answer:"""
        
        return prompt
    
    def answer_question(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Answer a question using retrieved context"""
        prompt = self.create_rag_prompt(query, context_chunks)
        response = self.generate_response(prompt)
        return response
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_name": self.config.llm_model,
            "device": str(self.device),
            "dtype": str(self.model.dtype),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "model_size_mb": sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        }


class QuantizedLLMInference:
    """Alternative implementation for quantized models (requires additional setup)"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load_quantized_model(self, model_name: str = None) -> None:
        """Load a quantized model for better CPU performance"""
        if model_name is None:
            model_name = self.config.quantized_llm_model
        
        try:
            logger.info(f"Loading quantized model: {model_name}")
            
            # This would require additional setup for GGML/GGUF models
            # For now, we'll use the standard transformers approach
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir
            )
            
            # Load with 8-bit quantization if available
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="cpu",
                torch_dtype=torch.float16,
                cache_dir=self.config.cache_dir
            )
            
            logger.info("Quantized model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load quantized model: {e}")
            # Fallback to regular model
            regular_llm = LocalLLMInference(self.config)
            regular_llm.load_model()
            self.model = regular_llm.model
            self.tokenizer = regular_llm.tokenizer