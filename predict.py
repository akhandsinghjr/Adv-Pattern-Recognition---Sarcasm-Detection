import torch
from transformers import BertTokenizer, BertForSequenceClassification
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# --- 1. Define Model Path and Prediction Function ---

# This MUST match the path you used in your notebook
MODEL_DIR = "./sarcasm_bert_model"

def predict_sarcasm_bert(comment, context, model, tokenizer, device):
    """
    Analyzes a new comment/context pair using the loaded BERT model.
    """
    print(f"\n--- Analyzing ---")
    print(f"Comment (C): {comment}")
    print(f"Context (X): {context}")
    
    # Format and tokenize
    input_text = comment + " [SEP] " + context
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)
    
    # Move tensors to the correct device (GPU/CPU)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        
    prediction = torch.argmax(outputs.logits, dim=1).cpu().item()
    
    # Print result
    result = "Offensive (Sarcastic)" if prediction == 1 else "Non-Offensive (Genuine)"
    print(f"--- Model Prediction: {result} ---")

# --- 2. Main execution (Loads model and starts query loop) ---
def main():
    print("--- Sarcasm Detection Inference Script ---")
    
    # Check if model directory exists
    if not os.path.exists(MODEL_DIR):
        print(f"\n--- FATAL ERROR ---")
        print(f"Model directory not found at: {MODEL_DIR}")
        print("Please make sure you have run the 'Save Model' cell in your 'aprasss.ipynb' notebook first.")
        return # Exit the script

    try:
        # 1. Load the tokenizer and model
        print(f"Loading model from {MODEL_DIR}...")
        tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
        model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
        
        # 2. Set up device (GPU or CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print(f"Model loaded successfully on {device}.")

    except Exception as e:
        print(f"\n--- FATAL ERROR ---")
        print(f"Could not load model. Error: {e}")
        return # Exit the script

    # 3. Start the query loop
    print("\nEnter a comment and its context. Type 'quit' for either to exit.")
    while True:
        comment = input("\nEnter Comment: ")
        if comment.lower() == 'quit':
            print("Exiting...")
            break
            
        context = input("Enter Context: ")
        if context.lower() == 'quit':
            print("Exiting...")
            break
            
        # 4. Get prediction
        predict_sarcasm_bert(comment, context, model, tokenizer, device)

if __name__ == "__main__":
    main()