import pandas as pd
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import os

def load_and_prepare_data(csv_path):
    """
    Load WhatsApp chat data and prepare conversation pairs
    """
    print("📂 Loading chat data...")
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    # Remove missing messages
    df = df.dropna(subset=['message'])
    
    # Clean up messages (remove very short/empty messages)
    df = df[df['message'].str.len() > 1]
    
    print(f"✅ Loaded {len(df)} messages")
    
    # Get unique senders
    senders = df['sender'].unique()
    print(f"👥 Found {len(senders)} participants: {senders}")
    
    # Create conversation pairs (prev_message -> response)
    conversations = []
    
    # Sort by timestamp to maintain conversation flow
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Create pairs where Romeo responds to the other person
    # This will train the bot to respond like Romeo
    prev_message = None
    prev_sender = None
    
    for _, row in df.iterrows():
        current_sender = row['sender']
        current_message = str(row['message']).strip()
        
        # If previous message was from the other person and current is from Romeo
        # This creates "question -> answer" pairs
        if prev_sender is not None:
            if prev_sender != current_sender and current_sender == 'Romeo':
                conversations.append([prev_message, current_message])
        
        prev_message = current_message
        prev_sender = current_sender
    
    print(f"📝 Created {len(conversations)} conversation pairs")
    print("\nExample conversation pair:")
    if conversations:
        print(f"  Q: {conversations[0][0]}")
        print(f"  A: {conversations[0][1]}")
    
    return conversations

def train_chatbot(conversations):
    """
    Train the chatbot using the conversation data
    """
    print("\n🤖 Initializing chatbot...")
    
    # Create chatbot instance
    chatbot = ChatBot(
        'RomeoBot',
        storage_adapter='chatterbot.storage.SQLStorageAdapter',
        database_uri='sqlite:///whatsapp_chatbot.db',
        logic_adapters=[
            {
                'import_path': 'chatterbot.logic.BestMatch',
                'default_response': "I don't have a response for that. Tell me more?",
                'maximum_similarity_threshold': 0.90
            },
            'chatterbot.logic.MathematicalEvaluation'
        ],
        preprocessors=[
            'chatterbot.preprocessors.clean_whitespace'
        ]
    )
    
    # Clear any existing data
    chatbot.storage.drop()
    
    print("📚 Training on conversation data...")
    
    # Train the bot
    trainer = ListTrainer(chatbot)
    
    # Train on each conversation pair
    for pair in conversations:
        trainer.train(pair)
    
    print("✅ Training complete!")
    print(f"🎯 Bot is ready to chat like Romeo!")
    
    return chatbot

def main():
    # Path to your CSV file
    CSV_PATH = 'cleaned_whatsapp_chat11.csv'
    
    # Check if file exists
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: {CSV_PATH} not found!")
        return
    
    # Load data and train
    conversations = load_and_prepare_data(CSV_PATH)
    
    if len(conversations) < 10:
        print("⚠️ Warning: Very few conversation pairs. Bot may not respond well.")
    
    # Train the bot
    bot = train_chatbot(conversations)
    
    print("\n" + "="*50)
    print("🎉 TRAINING COMPLETE!")
    print("Run 'python chat.py' to start chatting")
    print("="*50)

if __name__ == "__main__":
    main()