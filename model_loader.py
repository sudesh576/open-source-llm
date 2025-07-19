"""
Simple model loader for local Hugging Face models
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

class LocalModelLoader:
    def __init__(self, local_dir="./models/qwen-model"):
        self.local_dir = local_dir
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.tokenizer = None
        self.model = None
    
    def load_model(self):
        # Download if not exists
        if not os.path.exists(self.local_dir):
            print("Downloading model...")
            snapshot_download(
                repo_id=self.model_name,
                local_dir=self.local_dir,
                local_dir_use_symlinks=False
            )
        
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir)
        self.model = AutoModelForCausalLM.from_pretrained(self.local_dir)
        
        return self.tokenizer, self.model
    
    def generate(self, prompt, max_length=100):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response