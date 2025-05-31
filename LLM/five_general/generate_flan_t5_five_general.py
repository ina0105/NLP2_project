from transformers import T5TokenizerFast, T5ForConditionalGeneration
import torch
import numpy as np
import json
from words import words  
def get_multiple_context_embeddings(words,model_name="google/flan-t5-xl"):
    # tokenizer and model
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    embeddings = {}
    for word in words:
        word_embeddings = []
        contexts = [f"The topic was about {word}",
                f"They were discussing the {word} in the meeting",
                f"The {word} was the main focus of the conversation",
                f"During the lecture, they mentioned the {word}",
                f"The article discussed various aspects of {word}"] 
        for context in contexts:        
        # tokenizing the context
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
                    continue  # this is for special tokens
                if (start <= word_start < end) or (start < word_end <= end) or (word_start <= start and end <= word_end):
                    word_token_indices.append(i)
            inputs = {k: v.to(device) for k, v in tokens.items() if k != 'offset_mapping'}
            with torch.no_grad():
                outputs = model.encoder(**inputs)
                hidden_states = outputs.last_hidden_state.squeeze(0)  
            if word_token_indices:
                # embeddings for the word tokens
                word_embedding = hidden_states[word_token_indices].mean(dim=0)
                print("managed to find the word!")
            else:
                print(f"Word '{word}' not found in tokenized output. Using sentence mean as fallback.")
                word_embedding = hidden_states.mean(dim=0)
            word_embedding = word_embedding.detach().cpu()
            print(f"Shape for '{word}' in context: {context} is {word_embedding.shape}")
            word_embeddings.append(word_embedding.numpy() if hasattr(word_embedding, 'numpy') else np.array(word_embedding))        # Average the embeddings from all contexts
        combined_embedding = np.mean(word_embeddings, axis=0) #mean across contexts
        embeddings[word] = combined_embedding.tolist()
    
    return embeddings

def main():
    print("Generating embeddings from multiple contexts...")
    embeddings = get_multiple_context_embeddings(words)
    
    # saving embeddings 
    with open("flan_t5_five_general.json", "w") as f:
        json.dump(embeddings, f)
    
    print(f"Embeddings generated and saved to flan_t5_five_general.json")
    print(f"Number of words processed: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[list(embeddings.keys())[0]])}")

if __name__ == "__main__":
    main() 