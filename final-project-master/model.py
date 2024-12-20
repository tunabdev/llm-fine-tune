import json
import torch
from datasets import Dataset, DatasetDict
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

def load_and_split_dataset(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    dataset = Dataset.from_list(data)
    dataset = dataset.train_test_split(test_size=0.2)
    validation_test_split = dataset["test"].train_test_split(test_size=0.5)
    dataset["validation"] = validation_test_split["train"]
    dataset["test"] = validation_test_split["test"]

    return DatasetDict({
        "train": dataset["train"],
        "validation": dataset["validation"],
        "test": dataset["test"]
    })

def preprocess_function(examples):
    return {
        "text": [q + " " + a for q, a in zip(examples["question"], examples["answer"])]
    }

def fine_tune_gpt2(dataset_path):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Kullanılan cihaz:", device)
    model = model.to(device)
    
    datasets = load_and_split_dataset(dataset_path)
    tokenized_datasets = datasets.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=128), batched=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=500,
        eval_steps=500,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=50,
        push_to_hub=False,
    )

    print(f"Trainer is using device: {training_args.device}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    print("Model eğitimi tamamlandı.")

    model.save_pretrained("./fine_tuned_gpt2")
    tokenizer.save_pretrained("./fine_tuned_gpt2")

    print("Model kaydedildi.")

    results = trainer.evaluate()
    print("Evaluation Results:", results)

if __name__ == "__main__":
    dataset_path = "sets/dataset_augmented_single_per_disease.json"
    fine_tune_gpt2(dataset_path)
