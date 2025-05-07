import os
import torch
from extract_blip2_representations import extract_representations_for_word

IMAGE_ROOT = "/home/scur2169/NLP2_project/images"
OUTPUT_DIR = "/home/scur2169/NLP2_project/vlm_representations"

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

os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_valid_image(file):
    return file.lower().endswith((".jpg", ".jpeg", ".png"))

for word in words:
    folder_name = word.capitalize()
    image_dir = os.path.join(IMAGE_ROOT, folder_name)

    if not os.path.isdir(image_dir):
        print(f"Skipping {word} — folder not found: {image_dir}")
        continue

    image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) if is_valid_image(f)]

    if len(image_paths) < 6:
        print(f"Skipping {word} — only found {len(image_paths)} images.")
        continue

    print(f"Processing word: {word}")
    try:
        reps = extract_representations_for_word(word, image_paths)
        torch.save(reps, os.path.join(OUTPUT_DIR, f"{word}.pt"))
    except Exception as e:
        print(f"Failed to process {word}: {e}")
