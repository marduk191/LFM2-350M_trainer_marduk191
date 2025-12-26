import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./lfm2-350m-zimage-turbo"

def generate_prompt(user_input):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )

    instruction = "Transform the following image description into a detailed prompt for Z-Image Turbo. Use rich details, lighting info, and camera specs. Do not include negative prompts."
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{user_input}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:\n")[-1].strip()

if __name__ == "__main__":
    test_input = "A futuristic samurai in neon Tokyo"
    print(f"Input: {test_input}")
    print("-" * 30)
    # Note: Requires training to be completed first
    try:
        result = generate_prompt(test_input)
        print(f"Generated Z-Image Turbo Prompt:\n{result}")
    except Exception as e:
        print(f"Error (likely model not trained yet): {e}")
