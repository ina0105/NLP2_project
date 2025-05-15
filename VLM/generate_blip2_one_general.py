import os
import torch
from extract_blip2_representations import extract_representations_from_sentences

IMAGE_ROOT = "/home/scur2169/NLP2_project/images"
OUTPUT_DIR = "/home/scur2169/NLP2_project/vlm_one_general_representations"

# Template sentence
TEMPLATE = "They were thinking about the {} as they walked through the city."

# List of target words
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

def is_valid_image(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png"))

for word in words:
    folder = os.path.join(IMAGE_ROOT, word.capitalize())

    if not os.path.isdir(folder):
        print(f"Skipping {word} — folder not found: {folder}")
        continue

    image_paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if is_valid_image(f)]

    if len(image_paths) < 6:
        print(f"Skipping {word} — only found {len(image_paths)} images.")
        continue

    print(f"Processing word: {word}")
    try:
        sentence = TEMPLATE.format(word)
        reps = extract_representations_from_sentences(word, image_paths, [sentence], format_sentences=False)
        torch.save(reps, os.path.join(OUTPUT_DIR, f"{word}.pt"))
    except Exception as e:
        print(f"Failed to process {word}: {e}")
