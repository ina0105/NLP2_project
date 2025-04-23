from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np
import json

def get_flan_t5_embeddings(words, model_name="google/flan-t5-xl"):
    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    embeddings = {}
    
    for word in words:
        # Tokenize the word
        inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get encoder outputs
        with torch.no_grad():
            outputs = model.encoder(**inputs)
        
        # Get the embedding (using the last hidden state)
        word_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
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
    
    print("Generating embeddings...")
    embeddings = get_flan_t5_embeddings(words)
    
    # Save embeddings to a JSON file
    with open("output/flan_t5_one_word.json", "w") as f:
        json.dump(embeddings, f)
    
    
    print(f"Embeddings generated and saved to flan_t5_one_word.json")
    print(f"Number of words processed: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[words[0]])}")

if __name__ == "__main__":
    main() 