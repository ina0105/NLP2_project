from transformers import T5TokenizerFast, T5ForConditionalGeneration
import torch
import numpy as np
import json
from contexts import CONTEXTS

def get_multiple_context_embeddings(model_name="google/flan-t5-xl"):
    # Load tokenizer and model
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    embeddings = {}
    
    for word, word_contexts in CONTEXTS.items():
        word_embeddings = []
        
        for context in word_contexts:
            tokens = tokenizer(context, return_tensors="pt", return_offsets_mapping=True, padding=True, truncation=True)
            input_ids = tokens["input_ids"][0]
            offsets = tokens["offset_mapping"][0]
            context_lower = context.lower()
            word_lower = word.lower()
            word_start = context_lower.find(word_lower)
            word_end = word_start + len(word_lower)
            word_token_indices = []
            for i, (start, end) in enumerate(offsets.tolist()):
                if start == 0 and end == 0:
                    continue  # Special tokens
                # Check if token span overlaps with word span
                if (start <= word_start < end) or (start < word_end <= end) or (word_start <= start and end <= word_end):
                    word_token_indices.append(i)
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in tokens.items() if k != 'offset_mapping'}
            with torch.no_grad():
                outputs = model.encoder(**inputs)
                hidden_states = outputs.last_hidden_state.squeeze(0)  # shape: (seq_len, hidden_dim)
            #ovo izbaci
            # Print debug info
            print(f"\nContext: {context}")
            print(f"Word: '{word}' (span: {word_start}-{word_end})")
            print("Tokens and offsets:")
            for i, (token_id, (start, end)) in enumerate(zip(input_ids, offsets.tolist())):
                token_text = tokenizer.decode([token_id])
                print(f"  Token {i}: '{token_text}' (span: {start}-{end})")
            print(f"Matched token indices for '{word}': {word_token_indices}")
            if word_token_indices:
                matched_tokens = [tokenizer.decode([input_ids[i]]) for i in word_token_indices]
                print(f"Matched token texts: {matched_tokens}")
            else:
                print("No tokens matched; using sentence mean.")
            #ovo izbaci
            if word_token_indices:
                # Get the embeddings for the word tokens
                word_embedding = hidden_states[word_token_indices].mean(dim=0)
                print("managed to find the word!")
            else:
                print(f"Word '{word}' not found in tokenized output. Using sentence mean as fallback.")
                word_embedding = hidden_states.mean(dim=0)
            word_embedding = word_embedding.detach().cpu()
            print(f"Shape for '{word}' in context: {context} is {word_embedding.shape}")
            word_embeddings.append(word_embedding.numpy() if hasattr(word_embedding, 'numpy') else np.array(word_embedding)) 
        
        # Average the embeddings from all contexts
        combined_embedding = np.mean(word_embeddings, axis=0)
        embeddings[word] = combined_embedding.tolist()
    
    return embeddings

def main():
    print("Generating embeddings from multiple contexts...")
    embeddings = get_multiple_context_embeddings()
    
    # Save embeddings to a JSON file
    with open("flan_t5_five_distinct.json", "w") as f:
        json.dump(embeddings, f)
    
    print(f"Embeddings generated and saved to flan_t5_five_distinct.json")
    print(f"Number of words processed: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[list(embeddings.keys())[0]])}")

if __name__ == "__main__":
    main() 