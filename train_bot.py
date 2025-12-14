import pandas as pd
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer, ChatterBotCorpusTrainer
import os
import yaml
import glob  # For finding all CSV files

def load_and_prepare_data(directory_path):
    """
    Load ALL CSV files from a directory and prepare conversation pairs
    """
    print(f"📂 Loading CSV files from {directory_path}...")
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found in {directory_path}")
        return []
    
    print(f"📄 Found {len(csv_files)} CSV files")
    
    all_conversations = []
    total_messages = 0
    
    # Process each file
    for i, file_path in enumerate(csv_files, 1):
        filename = os.path.basename(file_path)
        print(f"   [{i}/{len(csv_files)}] Processing {filename}...")
        
        try:
            # Load CSV
            df = pd.read_csv(file_path)
            
            # Clean data
            df = df.dropna(subset=['message'])
            df = df[df['message'].str.len() > 1]
            
            # Convert timestamp and sort
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            total_messages += len(df)
            
            # Extract conversation pairs
            prev_message = None
            prev_sender = None
            
            for _, row in df.iterrows():
                current_sender = row['sender']
                current_message = str(row['message']).strip()
                
                if prev_sender is not None:
                    if prev_sender != current_sender and current_sender == 'Romeo':
                        all_conversations.append([prev_message, current_message])
                
                prev_message = current_message
                prev_sender = current_sender
                
        except Exception as e:
            print(f"⚠️  Error processing {filename}: {e}")
            continue
    
    print(f"✅ Loaded {total_messages} total messages")
    print(f"👥 Created {len(all_conversations)} conversation pairs across all files")
    
    if all_conversations:
        print("\nExample conversation pair:")
        print(f"  Q: {all_conversations[0][0]}")
        print(f"  A: {all_conversations[0][1]}")
    
    return all_conversations

def load_custom_corpus(file_path):
    """Load custom YAML responses"""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data.get('conversations', [])
    return []

def train_chatbot(conversations, custom_corpus_path='custom_responses.yml'):
    print("\n🤖 Initializing RomeoBot...")
    
    chatbot = ChatBot(
        'RomeoBot',
        storage_adapter='chatterbot.storage.SQLStorageAdapter',
        database_uri='sqlite:///whatsapp_chatbot.db',
        logic_adapters=[
            {
                'import_path': 'chatterbot.logic.BestMatch',
                'default_response': "I don't have a response for that. Tell me more? 🤔",
                'maximum_similarity_threshold': 0.90
            },
            'chatterbot.logic.MathematicalEvaluation'
        ],
        preprocessors=[
            'chatterbot.preprocessors.clean_whitespace'
        ]
    )
    
    chatbot.storage.drop()
    
    # === MULTI-SOURCE TRAINING ===
    
    # 1. Train on all WhatsApp conversations
    print("📱 Training on WhatsApp conversations...")
    list_trainer = ListTrainer(chatbot)
    
    # Train in batches for better performance
    batch_size = 100
    for i in range(0, len(conversations), batch_size):
        batch = conversations[i:i+batch_size]
        for pair in batch:
            list_trainer.train(pair)
        
        if i % 500 == 0:
            print(f"   ➡️ Trained {i}/{len(conversations)} pairs...")
    
    # 2. Train on custom corpus
    if os.path.exists(custom_corpus_path):
        print("📚 Training on custom responses...")
        custom_conversations = load_custom_corpus(custom_corpus_path)
        for pair in custom_conversations:
            list_trainer.train(pair)
    
    # 3. Train on general knowledge
    print("🌍 Training on general knowledge corpora...")
    corpus_trainer = ChatterBotCorpusTrainer(chatbot)
    corpus_trainer.train(
        'chatterbot.corpus.english.greetings',
        'chatterbot.corpus.english.conversations'
    )
    
    print("✅ Training complete!")
    return chatbot

def main():
    # Use DIRECTORY instead of single file
    DIRECTORY_PATH = 'chats/'  # Folder containing all CSVs
    
    if not os.path.exists(DIRECTORY_PATH):
        print(f"❌ Error: Directory '{DIRECTORY_PATH}' not found!")
        print("Please create the folder and add your CSV files.")
        return
    
    conversations = load_and_prepare_data(DIRECTORY_PATH)
    
    if len(conversations) < 10:
        print("⚠️  Warning: Very few conversation pairs found. Check your CSV files.")
        return
    
    bot = train_chatbot(conversations)
    
    print("\n" + "="*50)
    print("🎉 TRAINING COMPLETE!")
    print(f"🎯 Bot trained on {len(conversations)} conversation pairs")
    print("Run 'python chat.py' to start chatting")
    print("="*50)

if __name__ == "__main__":
    main()