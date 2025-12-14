import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os

class RomeoAI:
    def __init__(self, model_path="./romeo_gpt2_model"):
        print("🤖 Loading Romeo AI...")
        
        # Load the fine-tuned model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Set pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Load conversation history for context
        self.conversation_history = []
        
    def generate_response(self, user_input, max_length=80, temperature=0.8):
        """Generate a response using the fine-tuned model"""
        
        # Add to conversation history
        self.conversation_history.append(f"User: {user_input}")
        
        # Keep only last 3 exchanges for context
        if len(self.conversation_history) > 6:
            self.conversation_history = self.conversation_history[-6:]
        
        # Format prompt
        prompt = "\n".join(self.conversation_history) + "\nRomeo:"
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new generated text
        response = response[len(prompt):].strip()
        
        # Clean up (stop at next "User:" or newline)
        response = response.split("\n")[0].split("User:")[0].strip()
        
        # Add to history
        if response:
            self.conversation_history.append(f"Romeo: {response}")
        
        return response
    
    def chat(self):
        """Interactive chat loop"""
        print("\n" + "="*60)
        print("💬 CHAT WITH ROMEO AI (GPT-2)")
        print("="*60)
        print("Type 'quit' or 'exit' to stop")
        print("Type 'clear' to reset conversation history")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("🗑️ Conversation history cleared!\n")
                    continue
                
                if not user_input:
                    continue
                
                print("Romeo is typing...")
                response = self.generate_response(user_input)
                print(f"Romeo: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                break

def main():
    # Check if model exists
    if not os.path.exists("./romeo_gpt2_model/config.json"):
        print("❌ Model not found! Run 'python train_transformers.py' first.")
        return
    
    ai = RomeoAI()
    ai.chat()

if __name__ == "__main__":
    main()