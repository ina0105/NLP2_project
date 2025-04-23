from extract_blip2_representations import extract_representations_for_word

# Example image paths for the word "beer"
image_paths = [
    "/home/scur2169/NLP2_project/images/Beer/beer_1.jpg",
    "/home/scur2169/NLP2_project/images/Beer/beer_2.jpg",
    "/home/scur2169/NLP2_project/images/Beer/beer_3.jpg",
    "/home/scur2169/NLP2_project/images/Beer/beer_4.jpg",
    "/home/scur2169/NLP2_project/images/Beer/beer_5.jpg",
    "/home/scur2169/NLP2_project/images/Beer/beer_6.jpg",
]

word = "beer"
reps = extract_representations_for_word(word, image_paths)

for template, vector in reps.items():
    print(f"Template: {template}")
    print(f"Vector shape: {vector.shape}")
