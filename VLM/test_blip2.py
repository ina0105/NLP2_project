from transformers import BitsAndBytesConfig
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


#bnb_config = BitsAndBytesConfig(load_in_8bit=True)

# Load processor and model from local cache
model_path = "/home/scur2169/NLP2_project/models/models--Salesforce--blip2-flan-t5-xl/snapshots/0eb0d3b46c14c1f8c7680bca2693baafdb90bb28"
processor = Blip2Processor.from_pretrained(model_path)
model = Blip2ForConditionalGeneration.from_pretrained(model_path,torch_dtype = torch.bfloat16).to(device)

# Path to your image (adjust as needed)
image_path = "/home/scur2169/NLP2_project/images/Beer/beer_1.jpg"

# Load the image
image = Image.open(image_path).convert("RGB")
text = "Describe this image."

# Preprocess
inputs = processor(images=image, text=text, return_tensors="pt").to(device, dtype = torch.bfloat16)

generated_ids = model.generate(**inputs)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("✅ Model ran successfully!")
print("Generated text:", generated_text)
