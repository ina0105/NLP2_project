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


def extract_word_token_indices_v2(word: str, text: str, tokenizer) -> List[int]:
    tokens = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, padding=True, truncation=True)
    input_ids = tokens["input_ids"][0]
    offsets = tokens["offset_mapping"][0]

    context_lower = text.lower()
    word_lower = word.lower()
    word_start = context_lower.find(word_lower)
    word_end = word_start + len(word_lower)

    matched_indices = []
    for i, (start, end) in enumerate(offsets.tolist()):
        if start == 0 and end == 0:
            continue  # Special token
        if (start <= word_start < end) or (start < word_end <= end) or (word_start <= start and end <= word_end):
            matched_indices.append(i)

    return matched_indices


def extract_representation_from_image_text(image_path: str, text: str, word: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=text, return_tensors="pt").to(device, dtype=torch.bfloat16)

    with torch.no_grad():
        encoder_outputs = model.language_model.encoder(input_ids=inputs.input_ids,attention_mask=inputs.attention_mask,output_hidden_states=True,return_dict=True)
        last_hidden = encoder_outputs.hidden_states[-1].squeeze(0)

    input_ids = inputs.input_ids
    offset_mapping = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    token_indices = extract_word_token_indices_v2(word, text, tokenizer)

    if not token_indices:
        raise ValueError(f"Could not find word '{word}' in tokenized sequence: {tokenizer.convert_ids_to_tokens(input_ids.squeeze())}")

    word_representation = last_hidden[token_indices].mean(dim=0)  # Average across tokens if split
    return word_representation  # shape: (hidden_dim,)

def extract_representations_from_sentences(
    word: str,
    image_paths: List[str],
    sentences: List[str],
    format_sentences: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Extract averaged image-grounded word representations for a list of sentence contexts.

    Args:
        word (str): The target word to extract representations for.
        image_paths (List[str]): Paths to images associated with the word.
        sentences (List[str]): Either raw sentences (custom contexts) or templates to be formatted.
        format_sentences (bool): If True, format each sentence with the word.

    Returns:
        Dict[str, torch.Tensor]: Mapping from context sentence to averaged representation.
    """
    reps_by_context = {}

    text_variants = [s.format(word) if format_sentences else s for s in sentences]

    for text in text_variants:
        reps = []
        for image_path in image_paths:
            rep = extract_representation_from_image_text(image_path, text, word)
            reps.append(rep)
        reps_by_context[text] = torch.stack(reps).mean(dim=0)

    return reps_by_context
