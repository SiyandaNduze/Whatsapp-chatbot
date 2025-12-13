from chatterbot import ChatBot
from chatterbot.conversation import Statement
import sys

def load_chatbot():
    """
    Load the trained chatbot from the database
    """
    print("🤖 Loading RomeoBot...")
    
    chatbot = ChatBot(
        'RomeoBot',
        storage_adapter='chatterbot.storage.SQLStorageAdapter',
        database_uri='sqlite:///whatsapp_chatbot.db',
        logic_adapters=[
            {
                'import_path': 'chatterbot.logic.BestMatch',
                'default_response': "I don't have a response for that. Tell me more about yourself? 🤔",
                'maximum_similarity_threshold': 0.90
            }
        ]
    )
    
    return chatbot

def chat_with_bot():
    """
    Interactive chat function
    """
    bot = load_chatbot()
    
    print("\n" + "="*60)
    print("💬 CHAT WITH ROMEOBOT")
    print("="*60)
    print("Type 'quit' or 'exit' to stop chatting")
    print("Type 'clear' to clear the screen")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                print("\n" * 50)  # Simple clear screen
                continue
            
            if not user_input:
                continue
            
            # Get bot response
            response = bot.get_response(user_input)
            print(f"RomeoBot: {response}\n")
            
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            break

def chat_once(message):
    """
    Get a single response (useful for testing)
    """
    bot = load_chatbot()
    response = bot.get_response(message)
    return str(response)

if __name__ == "__main__":
    chat_with_bot()