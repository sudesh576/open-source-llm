#!/usr/bin/env python3
"""
Simple main.py for AI text generation
"""

from src.model_loader import LocalModelLoader

def main():
    # Setup
    model_path = "./models/qwen-model"
    loader = LocalModelLoader(model_path)
    
    print("Loading model...")
    tokenizer, model = loader.load_model()
    
    if not tokenizer or not model:
        print("Failed to load model!")
        return
    
    print("Model loaded! Type 'quit' to exit.\n")
    
    # Simple chat loop
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit']:
            break
        
        if not user_input.strip():
            continue
            
        # Generate response
        response = loader.generate(user_input, max_length=100)
        
        # Show only the new generated text
        new_text = response[len(user_input):].strip()
        print(f"AI: {new_text}\n")

if __name__ == "__main__":
    main()