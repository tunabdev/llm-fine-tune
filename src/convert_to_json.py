import pandas as pd
import json

df = pd.read_csv('Final_Augmented_dataset_Diseases_and_Symptoms.csv')

data = []
for index, row in df.iterrows():
    disease = row['diseases']
    symptoms = []

    for symptom, value in row.items():
        if symptom != 'diseases' and value == 1:
            symptoms.append(symptom)
    
    symptoms_text = ", ".join(symptoms)
    question = f"Here are my symptoms: {symptoms_text}. What could it mean?"
    answer = f"This may indicate {disease}."
    
    data.append({
        "question": question,
        "answer": answer
    })

with open('dataset_formatted.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)
