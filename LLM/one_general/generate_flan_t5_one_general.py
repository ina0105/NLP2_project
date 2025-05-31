from transformers import T5TokenizerFast, T5ForConditionalGeneration
import torch
import numpy as np
import json

def get_contextual_embeddings(words, model_name="google/flan-t5-xl"):
    # tokenizer and model
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    embeddings = {}
    
    for word in words:
        # simple context with the target word
        context = f"They were thinking about the {word} as they walked through the city"
        
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
    
    # saving embeddings 
    with open("output/flan_t5_one_general.json", "w") as f:
        json.dump(embeddings, f)
    
    print(f"Contextual embeddings generated and saved to flan_t5_one_general.json")
    print(f"Number of words processed: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[list(embeddings.keys())[0]])}")

if __name__ == "__main__":
    main() 