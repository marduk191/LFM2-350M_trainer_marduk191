import json
import random
import os

# Templates for Z-Image Turbo prompts: [Shot & subject] + [Age & appearance] + [Clothing] + [Environment] + [Lighting] + [Mood] + [Style] + [Technical]
SUBJECTS = [
    "A majestic lion", "A futuristic city", "A cyberpunk girl", "An elderly wizard", 
    "A cozy cottage", "A sleek sports car", "An astronaut", "A mystical forest",
    "A high-end watch", "A delicious sushi platter", "A steam-punk owl", "A serene beach",
    "A post-apocalyptic survivor", "A digital deity", "A classical ballerina", "A street food vendor",
    "A dragon made of ice", "A Victorian detective", "A robotic samurai", "An underwater coral temple",
    "A floating mountain landscape", "A group of nomadic traders", "A futuristic lab", "A haunted mansion",
    "A 1950s diner at night", "A colony on Mars", "A mystical portal in a cave", "A steampunk airship",
    "A zen garden with koi fish", "A busy marketplace in Marrakesh", "A snowy log cabin", "A desert oasis",
    "A phoenix rising from ashes", "A gothic cathedral", "A futuristic surgical robot", "A vintage typewriter",
    "A gladiator in a coliseum", "A futuristic greenhouse", "A crystal-clear waterfall", "A mystical owl with three eyes",
    "A cyberpunk hacker", "A medieval alchemist", "A futuristic train station", "A tranquil zen pagoda",
    "A bioluminescent deep-sea jellyfish", "A Victorian scientist", "A futuristic orbital station", "A nomadic desert caravan",
    "A massive stone golem", "A delicate glass flower", "A futuristic mecha pilot", "An ancient scroll with glowing runes"
]

STYLES = [
    "hyper-realistic photography", "cyberpunk aesthetic", "oil painting on canvas",
    "cinematic 35mm film", "minimalist digital art", "vaporwave style",
    "noir cinematography", "Studio Ghibli inspired anime", "surrealism", "baroque art",
    "macro photography", "street style photography", "editorial fashion shoot", "concept art",
    "double exposure", "long exposure", "infrared photography", "low poly 3d render",
    "pencil sketch", "watercolor painting", "3D isometric render", "glitch art", "pop art",
    "renaissance portrait", "tilt-shift photography", "fuji color film look", "kodak portra 400 aesthetic"
]

LIGHTING = [
    "soft diffused daylight", "moody purple and blue neon lights", "golden hour sunshine",
    "harsh dramatic shadows", "volumetric fog and god rays", "soft flickering candlelight",
    "bioluminescent glow", "harsh midday sun", "fluorescent lab lighting", "starlight and galaxy glow",
    "rim lighting", "split lighting", "cinematic teal and orange", "natural moonlight",
    "neon silhouette", "dappled sunlight through leaves", "soft studio lighting"
]

TECHNICAL = [
    "8K resolution, highly detailed, sharp focus, no blur, shot on Leica M6",
    "4K, masterpiece, intricate textures, shallow depth of field, 50mm lens",
    "unreal engine 5 render, global illumination, ray tracing, sharp details",
    "natural look, crisp edges, high contrast, vivid colors",
    "shot on Hasselblad, medium format, fine grain, professional color grading",
    "85mm portrait lens, f/1.8, bokeh, skin texture detail",
    "extreme close-up, sharp optics, ultra-high resolution",
    "wide-angle lens, deep depth of field, clear structural lines"
]

ENVIRONMENT = [
    "in a dense futuristic neon-lit Tokyo street", "at the edge of a floating island above clouds",
    "inside a cluttered library filled with glowing crystals", "on a desolate Martian landscape",
    "in a rainy London alleyway at night", "in a sun-drenched Mediterranean village",
    "buried deep within an ancient overgrown jungle", "surrounded by a sea of purple sand",
    "inside a luxurious penthouse with floor-to-ceiling windows", "at a crowded festival in India",
    "floating in the zero-gravity of deep space", "in a frozen arctic wasteland under aurora borealis",
    "on top of a crumbling skyscraper in a post-apocalyptic city", "inside a magical subterranean grotto",
    "at a quiet countryside train station at dawn", "in a high-tech server room with glowing wires"
]

MOODS = [
    "mysterious and atmospheric", "vibrant and energetic", "peaceful and serene",
    "melancholic and somber", "epic and powerful", "whimsical and playful",
    "tense and cinematic", "dreamy and ethereal", "calm and meditative", "intense and gritty",
    "ominous and dark", "bright and hopeful", "nostalgic and warm", "lonely and isolated"
]

def generate_z_image_prompt(base_subject):
    style = random.choice(STYLES)
    light = random.choice(LIGHTING)
    tech = random.choice(TECHNICAL)
    env = random.choice(ENVIRONMENT)
    mood = random.choice(MOODS)
    
    # Structure: Subject + Context + Style + Lighting + Mood + Technical
    prompt = f"{base_subject} {env}. {style}, {light}, {mood}, {tech}."
    return prompt

def create_dataset(count=60000):
    dataset = []
    print(f"Generating {count} prompts...")
    for i in range(count):
        if i % 10000 == 0 and i > 0:
            print(f"Generated {i} prompts...")
            
        base_subject = random.choice(SUBJECTS)
        # Add some variation to the base subject to make it look like user input
        variations = [
            base_subject,
            f"Show me {base_subject.lower()}",
            f"Generate a picture of {base_subject.lower()}",
            f"{base_subject.lower()}, high quality",
            f"A cooling drawing of {base_subject.lower()}",
            f"Prompt for {base_subject.lower()}",
            f"I want to see {base_subject.lower()} in 4k",
            f"Design {base_subject.lower()}",
            f"{base_subject} in a cool style",
            f"Z-Image prompt for {base_subject.lower()}"
        ]
        user_input = random.choice(variations)
        z_prompt = generate_z_image_prompt(base_subject)
        
        entry = {
            "instruction": "Transform the following image description into a detailed prompt for Z-Image Turbo. Use rich details, lighting info, and camera specs. Do not include negative prompts.",
            "input": user_input,
            "output": z_prompt
        }
        dataset.append(entry)
    
    # Ensure directory exists
    os.makedirs("training data", exist_ok=True)
    with open("training data/dataset.jsonl", "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Created training data/dataset.jsonl with {count} entries.")

if __name__ == "__main__":
    create_dataset(60000)
