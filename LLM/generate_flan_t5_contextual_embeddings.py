from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np
import json

def get_contextual_embeddings(words, model_name="google/flan-t5-xl"):
    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create contextual sentences for each word
    contexts = {
        "ability": "She demonstrated her ability to solve complex problems quickly.",
        "accomplished": "The accomplished musician performed a beautiful symphony.",
        "angry": "He was angry when he discovered the broken window.",
        "apartment": "They moved into a spacious apartment in the city center.",
        "applause": "The audience erupted in applause after the stunning performance.",
        "argument": "They had a heated argument about politics.",
        "argumentatively": "He spoke argumentatively during the debate.",
        "art": "The museum displayed beautiful art from various periods.",
        "attitude": "Her positive attitude helped the team succeed.",
        "bag": "She carried a heavy bag of groceries home.",
        "ball": "The children played with a colorful ball in the park.",
        "bar": "They met at the local bar for drinks.",
        "bear": "We saw a bear in the forest during our hike.",
        "beat": "The drummer kept a steady beat throughout the song.",
        "bed": "She made the bed with fresh sheets.",
        "beer": "He ordered a cold beer at the restaurant.",
        "big": "The big tree provided shade on hot days.",
        "bird": "A beautiful bird sang in the morning.",
        "blood": "The doctor analyzed the blood sample carefully.",
        "body": "Regular exercise keeps the body healthy.",
        "brain": "The brain processes information rapidly.",
        "broken": "The broken vase lay in pieces on the floor.",
        "building": "The new building towered over the city.",
        "burn": "The fire continued to burn throughout the night.",
        "business": "She started her own business last year.",
        "camera": "He captured the moment with his camera.",
        "carefully": "She carefully placed the fragile item in the box.",
        "challenge": "The difficult puzzle presented a real challenge.",
        "charity": "They donated money to a local charity.",
        "charming": "The charming village attracted many tourists.",
        "clothes": "She organized her clothes by color.",
        "cockroach": "A cockroach scurried across the kitchen floor.",
        "code": "The programmer wrote clean and efficient code.",
        "collection": "He showed us his impressive coin collection.",
        "computer": "The computer processed the data quickly.",
        "construction": "The construction of the bridge took two years.",
        "cook": "She loves to cook delicious meals for her family.",
        "counting": "The child was counting the stars in the sky.",
        "crazy": "The idea seemed crazy at first, but it worked.",
        "damage": "The storm caused significant damage to the roof.",
        "dance": "They performed a beautiful dance at the wedding.",
        "dangerous": "The mountain path was dangerous in winter.",
        "deceive": "He tried to deceive his friends with a false story.",
        "dedication": "Her dedication to the project was impressive.",
        "deliberately": "He deliberately ignored the warning signs.",
        "delivery": "The package arrived with the morning delivery.",
        "dessert": "They served a delicious chocolate dessert.",
        "device": "The new device made the task much easier.",
        "dig": "The archaeologist began to dig at the ancient site.",
        "dinner": "They prepared a special dinner for the guests.",
        "disease": "The doctor studied the rare disease carefully.",
        "dissolve": "The sugar will dissolve in hot water.",
        "disturb": "Please don't disturb me while I'm working.",
        "do": "What do you want to do this weekend?",
        "doctor": "The doctor examined the patient thoroughly.",
        "dog": "The dog wagged its tail happily.",
        "dressing": "She prepared a fresh salad dressing.",
        "driver": "The taxi driver knew the city well.",
        "economy": "The economy showed signs of improvement.",
        "election": "The election results were announced last night.",
        "electron": "The electron microscope revealed tiny details.",
        "elegance": "The ballroom was decorated with elegance.",
        "emotion": "The movie evoked strong emotion in the audience.",
        "emotionally": "She spoke emotionally about her experiences.",
        "engine": "The car's engine needed repair.",
        "event": "The charity event raised thousands of dollars.",
        "experiment": "The scientist conducted a careful experiment.",
        "extremely": "The weather was extremely hot today.",
        "feeling": "She had a strange feeling about the situation.",
        "fight": "The two boxers prepared for their fight.",
        "fish": "They caught several fish in the lake.",
        "flow": "The river's flow was strong after the rain.",
        "food": "The food at the restaurant was delicious.",
        "garbage": "Please take out the garbage before it smells.",
        "gold": "The ring was made of pure gold.",
        "great": "The concert was a great success.",
        "gun": "The police officer carried a gun for protection.",
        "hair": "She styled her hair for the special occasion.",
        "help": "Can you help me with this problem?",
        "hurting": "His back was hurting after the long drive.",
        "ignorance": "His ignorance of the facts was surprising.",
        "illness": "The illness kept her in bed for a week.",
        "impress": "The performance will impress the audience.",
        "invention": "The new invention revolutionized the industry.",
        "investigation": "The police conducted a thorough investigation.",
        "invisible": "The tiny particles were invisible to the naked eye.",
        "job": "He got a new job at the technology company.",
        "jungle": "They explored the dense jungle carefully.",
        "kindness": "Her kindness touched everyone she met.",
        "king": "The king ruled the kingdom wisely.",
        "lady": "The elegant lady entered the room gracefully.",
        "land": "The plane prepared to land at the airport.",
        "laugh": "The joke made everyone laugh.",
        "law": "The new law was passed by parliament.",
        "left": "She left the room quietly.",
        "level": "The water reached a dangerous level.",
        "liar": "He was known to be a compulsive liar.",
        "light": "The light from the window brightened the room.",
        "magic": "The magician performed amazing magic tricks.",
        "marriage": "Their marriage lasted fifty years.",
        "material": "The dress was made of expensive material.",
        "mathematical": "The problem required mathematical precision.",
        "mechanism": "The clock's mechanism needed repair.",
        "medication": "The doctor prescribed new medication.",
        "money": "They saved money for their future.",
        "mountain": "The mountain peak was covered in snow.",
        "movement": "The dance movement was graceful.",
        "movie": "The movie received excellent reviews.",
        "music": "The music filled the concert hall.",
        "nation": "The nation celebrated its independence.",
        "news": "The news spread quickly through the town.",
        "noise": "The noise from the construction was loud.",
        "obligation": "He felt an obligation to help his friend.",
        "pain": "The pain in her leg was unbearable.",
        "personality": "Her vibrant personality attracted many friends.",
        "philosophy": "He studied ancient philosophy in college.",
        "picture": "The picture captured a beautiful sunset.",
        "pig": "The farmer fed the pig in the morning.",
        "plan": "They made a detailed plan for the trip.",
        "plant": "The plant needed more sunlight to grow.",
        "play": "The children went outside to play.",
        "pleasure": "It was a pleasure to meet you.",
        "poor": "The poor family needed assistance.",
        "prison": "The prisoner escaped from the prison.",
        "professional": "She maintained a professional attitude.",
        "protection": "The law provided protection for workers.",
        "quality": "The product was known for its high quality.",
        "reaction": "His reaction to the news was surprising.",
        "read": "She loved to read books in the evening.",
        "relationship": "Their relationship grew stronger over time.",
        "religious": "The ceremony had religious significance.",
        "residence": "The president's residence was heavily guarded.",
        "road": "The road was closed for construction.",
        "sad": "The sad news affected everyone deeply.",
        "science": "The science experiment was successful.",
        "seafood": "The restaurant served fresh seafood.",
        "sell": "They decided to sell their old car.",
        "sew": "She learned to sew her own clothes.",
        "sexy": "The actor looked sexy in the new movie.",
        "shape": "The artist worked the clay into shape.",
        "ship": "The ship sailed across the ocean.",
        "show": "The magician put on an amazing show.",
        "sign": "The sign indicated the way to the exit.",
        "silly": "The children told silly jokes.",
        "sin": "The priest spoke about the concept of sin.",
        "skin": "The doctor examined the patient's skin.",
        "smart": "The smart student solved the problem quickly.",
        "smiling": "The smiling child ran to her mother.",
        "solution": "They found a creative solution to the problem.",
        "soul": "The music touched her soul deeply.",
        "sound": "The sound of the waves was calming.",
        "spoke": "He spoke clearly during the presentation.",
        "star": "The night sky was full of bright stars.",
        "student": "The student studied hard for the exam.",
        "stupid": "It was a stupid mistake to make.",
        "successful": "The business became very successful.",
        "sugar": "She added sugar to her coffee.",
        "suspect": "The police questioned the main suspect.",
        "table": "They gathered around the dinner table.",
        "taste": "The soup had a delicious taste.",
        "team": "The team worked together effectively.",
        "texture": "The fabric had a soft texture.",
        "time": "Time passed quickly during the vacation.",
        "tool": "He used the right tool for the job.",
        "toy": "The child played with his favorite toy.",
        "tree": "The old tree provided shade in summer.",
        "trial": "The court case went to trial.",
        "tried": "She tried her best to succeed.",
        "typical": "It was a typical summer day.",
        "unaware": "He was unaware of the danger.",
        "usable": "The old computer was still usable.",
        "useless": "The broken phone was now useless.",
        "vacation": "They planned a relaxing vacation.",
        "war": "The country was at war for many years.",
        "wash": "She needed to wash the dirty clothes.",
        "weak": "The bridge was too weak to cross.",
        "wear": "He decided to wear his new suit.",
        "weather": "The weather was perfect for a picnic.",
        "willingly": "She willingly helped her neighbor.",
        "word": "He couldn't find the right word to say."
    }
    
    embeddings = {}
    
    for word, context in contexts.items():
        # Tokenize the context
        inputs = tokenizer(context, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get encoder outputs
        with torch.no_grad():
            outputs = model.encoder(**inputs)
        
        # Get the embedding (using the last hidden state)
        word_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings[word] = word_embedding.tolist()
    
    return embeddings

def main():
    print("Generating contextual embeddings...")
    embeddings = get_contextual_embeddings([])  # Empty list since we're using predefined contexts
    
    # Save embeddings to a JSON file
    with open("output/flan_t5_embeddings.json", "w") as f:
        json.dump(embeddings, f)
    
    print(f"Contextual embeddings generated and saved to flan_t5_contextual_embeddings.json")
    print(f"Number of words processed: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[list(embeddings.keys())[0]])}")

if __name__ == "__main__":
    main() 