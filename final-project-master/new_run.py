import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_dir = "./fine_tuned_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

def generate_response(prompt, max_length=30):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    response = response.replace(prompt, "").strip()
    response = response.split(".")[0] + "."
    return response

if __name__ == "__main__":
    input_text = "I\u2019m noticing signs such as  involuntary urination, blood in urine, hand or finger stiffness or tightness, impotence, symptoms of bladder. What could it mean?. Should I be worried about something specific?"
    response = generate_response(input_text)
    print(f"Input: {input_text}")
    print(f"Response: {response}")