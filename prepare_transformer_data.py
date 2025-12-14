import pandas as pd
import json
import os
import glob

def prepare_conversational_data(directory_path, output_file='chat_data.json'):
    """
    Convert ALL CSV files to conversational format
    """
    print(f"🔄 Preparing transformer training data from {directory_path}...")
    
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found")
        return
    
    all_pairs = []
    total_files = 0
    
    for i, file_path in enumerate(csv_files, 1):
        filename = os.path.basename(file_path)
        print(f"   [{i}/{len(csv_files)}] Processing {filename}...")
        
        try:
            df = pd.read_csv(file_path)
            df = df.dropna(subset=['message'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Create conversation threads
            temp_convo = []
            for _, row in df.iterrows():
                temp_convo.append({
                    'sender': row['sender'],
                    'message': str(row['message']).strip(),
                    'time': row['timestamp']
                })
            
            # Group by time gaps
            current_convo = []
            for msg in temp_convo:
                if not current_convo:
                    current_convo.append(msg)
                else:
                    time_gap = (msg['time'] - current_convo[-1]['time']).total_seconds() / 60
                    if time_gap < 30:
                        current_convo.append(msg)
                    else:
                        # Extract pairs from completed conversation
                        for j in range(len(current_convo) - 1):
                            if current_convo[j]['sender'] != 'Romeo' and current_convo[j+1]['sender'] == 'Romeo':
                                all_pairs.append({
                                    "prompt": current_convo[j]['message'],
                                    "response": current_convo[j+1]['message']
                                })
                        current_convo = [msg]
            
            # Process last conversation
            for j in range(len(current_convo) - 1):
                if current_convo[j]['sender'] != 'Romeo' and current_convo[j+1]['sender'] == 'Romeo':
                    all_pairs.append({
                        "prompt": current_convo[j]['message'],
                        "response": current_convo[j+1]['message']
                    })
            
            total_files += 1
            
        except Exception as e:
            print(f"⚠️  Error processing {filename}: {e}")
            continue
    
    # Save all pairs
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_pairs, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Created {len(all_pairs)} total conversation pairs")
    print(f"📄 Saved to {output_file}")
    
    if all_pairs:
        print("\nExample:")
        print(f"Prompt: {all_pairs[0]['prompt']}")
        print(f"Response: {all_pairs[0]['response']}")

if __name__ == "__main__":
    prepare_conversational_data('chats/')