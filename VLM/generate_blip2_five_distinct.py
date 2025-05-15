from custom_context_templates_five import CONTEXTS
from extract_blip2_representations import extract_representations_from_sentences
import os
import torch

IMAGE_ROOT = "/home/scur2169/NLP2_project/images"
OUTPUT_DIR = "/home/scur2169/NLP2_project/vlm_five_distinct_representations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_valid_image(file):
    return file.lower().endswith((".jpg", ".jpeg", ".png"))

for word, contexts in CONTEXTS.items():
    folder = os.path.join(IMAGE_ROOT, word.capitalize())
    if not os.path.isdir(folder):
        print(f"Skipping {word}: no folder found")
        continue

    image_paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if is_valid_image(f)]
    if len(image_paths) < 6:
        print(f"Skipping {word}: not enough images")
        continue

    print(f"Processing word: {word}")
    try:
        reps = extract_representations_from_sentences(word, image_paths, CONTEXTS[word], format_sentences=False)
        torch.save(reps, os.path.join(OUTPUT_DIR, f"{word}.pt"))
    except Exception as e:
        print(f"Failed to process {word}: {e}")
