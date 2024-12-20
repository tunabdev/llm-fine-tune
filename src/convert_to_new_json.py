import json
import random
from collections import defaultdict

with open('dataset_formatted.json', 'r') as json_file:
    data = json.load(json_file)

disease_groups = defaultdict(list)
for entry in data:
    disease_groups[entry["answer"]].append(entry)

augmented_data = data.copy()
for disease, entries in disease_groups.items():
    selected_entry = random.choice(entries)
    original_question = selected_entry["question"]
    answer = selected_entry["answer"]

    variations = [
        f"I'm experiencing symptoms like {original_question[21:]}. Any thoughts on this?",
        f"These issues are affecting me: {original_question[21:]}. Do you know what this might mean?",
        f"My symptoms include {original_question[21:]}. Could this indicate something?",
        f"What could be the reason for these symptoms: {original_question[21:]}?",
        f"I’m noticing signs such as {original_question[21:]}. Should I be worried about something specific?"
    ]

    for variation in variations:
        augmented_data.append({
            "question": variation,
            "answer": answer
        })

with open('dataset_augmented_single_per_disease.json', 'w') as json_file:
    json.dump(augmented_data, json_file, indent=4)
