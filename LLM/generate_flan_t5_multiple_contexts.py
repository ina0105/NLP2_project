from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np
import json
from contexts import CONTEXTS

def get_multiple_context_embeddings(model_name="google/flan-t5-base"):
    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    embeddings = {}
    
    for word, word_contexts in CONTEXTS.items():
        word_embeddings = []
        
        for context in word_contexts:
            # Tokenize the context
            inputs = tokenizer(context, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get encoder outputs
            with torch.no_grad():
                outputs = model.encoder(**inputs)
            
            # Get the embedding (using the last hidden state)
            context_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            word_embeddings.append(context_embedding)
        
        # Average the embeddings from all contexts
        combined_embedding = np.mean(word_embeddings, axis=0)
        embeddings[word] = combined_embedding.tolist()
    
    return embeddings

def main():
    print("Generating embeddings from multiple contexts...")
    embeddings = get_multiple_context_embeddings()
    
    # Save embeddings to a JSON file
    with open("flan_t5_multiple_context_embeddings.json", "w") as f:
        json.dump(embeddings, f)
    
    print(f"Embeddings generated and saved to flan_t5_multiple_context_embeddings.json")
    print(f"Number of words processed: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[list(embeddings.keys())[0]])}")

if __name__ == "__main__":
    main() 