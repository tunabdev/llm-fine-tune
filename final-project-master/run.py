from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

model_path = "./fine_tuned_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Pipeline oluşturma
qa_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# Örnek input
examples = [
    "Here are my symptoms: anxiety and nervousness, shortness of breath, depressive or psychotic symptoms, chest tightness, palpitations, irregular heartbeat, breathing fast. What could it mean?"
]

for example in examples:
    print(f"Input: {example}")
    response = qa_pipeline(
        example,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        truncation = True,
    )
    print(f"Response: {response[0]['generated_text']}")
    print("-" * 50)
