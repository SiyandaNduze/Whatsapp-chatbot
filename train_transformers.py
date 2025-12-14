import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer
)
from datasets import load_dataset, Dataset
import json

def load_chat_dataset(json_path):
    """Load and format the conversation data"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Format: "User: {prompt}\nRomeo: {response}"
    formatted_data = []
    for pair in data:
        text = f"User: {pair['prompt']}\nRomeo: {pair['response']}"
        formatted_data.append(text)
    
    return Dataset.from_dict({"text": formatted_data})

def tokenize_function(examples, tokenizer):
    """Tokenize the conversations"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128  # Adjust based on your messages
    )

def main():
    # Configuration
    MODEL_NAME = "distilgpt2"  # Smaller, faster model
    OUTPUT_DIR = "./romeo_gpt2_model"
    DATA_PATH = "chat_data.json"
    
    print(f"🤖 Loading {MODEL_NAME}...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    # Load dataset
    print("📚 Loading chat dataset...")
    dataset = load_chat_dataset(DATA_PATH)
    
    # Tokenize
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    
    # Split into train/validation
    dataset = tokenized_dataset.train_test_split(test_size=0.1)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,  # Adjust higher for better results
        per_device_train_batch_size=4,  # Reduce if you get OOM errors
        per_device_eval_batch_size=4,
        eval_steps=100,
        save_steps=100,
        warmup_steps=50,
        logging_dir="./logs",
        logging_steps=10,
        learning_rate=5e-5,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=torch.cuda.is_available(),  # Speed up if GPU available
        report_to="none"  # Disable wandb for simplicity
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )
    
    print("🏋️ Starting fine-tuning...")
    trainer.train()
    
    # Save the model
    print(f"💾 Saving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("✅ Fine-tuning complete!")

if __name__ == "__main__":
    main()