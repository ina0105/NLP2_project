from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-xl"
cache_path = "/scratch-shared/scur2185/models/flan-t5-xl"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_path)
