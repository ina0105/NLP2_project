from transformers import T5TokenizerFast, T5ForConditionalGeneration
import torch
import numpy as np
import json

def get_contextual_embeddings(words, model_name="google/flan-t5-xl"):
    # Load tokenizer and model
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    embeddings = {}
    
    for word in words:
        # Create a simple context with the target word
        context = f"They were thinking about the {word} as they walked through the city"
        
        # Tokenize the context
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
        embeddings[word] = word_embedding.tolist()
    return embeddings

def main():
    words = [
        "ability", "accomplished", "angry", "apartment", "applause", "argument", "argumentatively", "art", "attitude",
        "bag", "ball", "bar", "bear", "beat", "bed", "beer", "big", "bird", "blood", "body", "brain", "broken",
        "building", "burn", "business", "camera", "carefully", "challenge", "charity", "charming", "clothes",
        "cockroach", "code", "collection", "computer", "construction", "cook", "counting", "crazy", "damage", "dance",
        "dangerous", "deceive", "dedication", "deliberately", "delivery", "dessert", "device", "dig", "dinner",
        "disease", "dissolve", "disturb", "do", "doctor", "dog", "dressing", "driver", "economy", "election",
        "electron", "elegance", "emotion", "emotionally", "engine", "event", "experiment", "extremely", "feeling",
        "fight", "fish", "flow", "food", "garbage", "gold", "great", "gun", "hair", "help", "hurting", "ignorance",
        "illness", "impress", "invention", "investigation", "invisible", "job", "jungle", "kindness", "king", "lady",
        "land", "laugh", "law", "left", "level", "liar", "light", "magic", "marriage", "material", "mathematical",
        "mechanism", "medication", "money", "mountain", "movement", "movie", "music", "nation", "news", "noise",
        "obligation", "pain", "personality", "philosophy", "picture", "pig", "plan", "plant", "play", "pleasure",
        "poor", "prison", "professional", "protection", "quality", "reaction", "read", "relationship", "religious",
        "residence", "road", "sad", "science", "seafood", "sell", "sew", "sexy", "shape", "ship", "show", "sign",
        "silly", "sin", "skin", "smart", "smiling", "solution", "soul", "sound", "spoke", "star", "student", "stupid",
        "successful", "sugar", "suspect", "table", "taste", "team", "texture", "time", "tool", "toy", "tree", "trial",
        "tried", "typical", "unaware", "usable", "useless", "vacation", "war", "wash", "weak", "wear", "weather",
        "willingly", "word"
    ]
    print("Generating contextual embeddings...")
    embeddings = get_contextual_embeddings(words)  
    
    # Save embeddings to a JSON file
    with open("output/flan_t5_one_general.json", "w") as f:
        json.dump(embeddings, f)
    
    print(f"Contextual embeddings generated and saved to flan_t5_one_general.json")
    print(f"Number of words processed: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[list(embeddings.keys())[0]])}")

if __name__ == "__main__":
    main() 