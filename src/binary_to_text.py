import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()


def main():
    
    df = pd.read_csv(os.getenv('DATASET_PATH'))
    disease_column = df.columns[0]
    symptom_columns = df.columns[1:]

    df['text'] = df.apply(lambda row: row_to_text(row, disease_column, symptom_columns), axis=1)
    df[['text']].to_csv('converted_dataset.csv', index=False)

def row_to_text(row, disease_column, symptom_columns):
    symptoms = [symptom for symptom in symptom_columns if row[symptom] == 1]
    
    if symptoms:
        symptoms_text = "these symptoms: " + ", ".join(symptoms)
    else:
        symptoms_text = "There is no symptom"

    disease = row[disease_column]
    
    return f"A person has {symptoms_text}. So this person has {disease}."


if __name__ == '__main__':
    main()