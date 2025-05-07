from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import PreTrainedTokenizerFast
from PIL import Image
import torch
import os
from typing import List, Tuple, Dict

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model & processor
model_path = "/home/scur2169/NLP2_project/models/models--Salesforce--blip2-flan-t5-xl/snapshots/0eb0d3b46c14c1f8c7680bca2693baafdb90bb28"
processor = Blip2Processor.from_pretrained(model_path)
model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
tokenizer: PreTrainedTokenizerFast = processor.tokenizer

TEMPLATE_SENTENCES= ["The topic was about {}",
                "They were discussing the {} in the meeting",
                "The {} was the main focus of the conversation",
                "During the lecture, they mentioned the {}",
                "The article discussed various aspects of {}"]

def extract_word_token_indices(word: str, input_ids: torch.Tensor, offsets: List[Tuple[int, int]]) -> List[int]:
    """
    Match the tokenized form of the word against the input token sequence and return its indices.
    """
    word_tokens = tokenizer(word, add_special_tokens=False).input_ids
    input_id_list = input_ids.squeeze().tolist()

    for i in range(len(input_id_list) - len(word_tokens) + 1):
        if input_id_list[i:i+len(word_tokens)] == word_tokens:
            return list(range(i, i + len(word_tokens)))

    return []  # if no match is found


def extract_representation_from_image_text(image_path: str, text: str, word: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=text, return_tensors="pt").to(device, dtype=torch.bfloat16)

    with torch.no_grad():
        encoder_outputs = model.language_model.encoder(input_ids=inputs.input_ids,attention_mask=inputs.attention_mask,output_hidden_states=True,return_dict=True)
        last_hidden = encoder_outputs.hidden_states[-1].squeeze(0)

    input_ids = inputs.input_ids
    offset_mapping = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    token_indices = extract_word_token_indices(word, input_ids, offset_mapping)

    if not token_indices:
        raise ValueError(f"Could not find word '{word}' in tokenized sequence: {tokenizer.convert_ids_to_tokens(input_ids.squeeze())}")

    word_representation = last_hidden[token_indices].mean(dim=0)  # Average across tokens if split
    return word_representation  # shape: (hidden_dim,)

def extract_representations_for_word(word: str, image_paths: List[str]) -> Dict[str, torch.Tensor]:
    """
    Return a dictionary mapping text variant to averaged representation over the 6 images.
    """
    text_variants = [template.format(word) for template in TEMPLATE_SENTENCES]
    reps_by_variant = {}

    for text in text_variants:
        reps = []
        for image_path in image_paths:
            rep = extract_representation_from_image_text(image_path, text, word)
            reps.append(rep)
        reps_by_variant[text] = torch.stack(reps).mean(dim=0)  # average over images
    return reps_by_variant
