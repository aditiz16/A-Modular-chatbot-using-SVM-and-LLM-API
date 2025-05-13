from classifier import predict_disease, get_valid_symptoms
import pandas as pd

import requests

HUGGINGFACE_API_KEY = "YOUR HUGGING FACE API KEY "
MODEL = "HuggingFaceH4/zephyr-7b-beta" 
# Load the datasets
description_df = pd.read_csv(r"MasterData\symptom_Description.csv")
precaution_df = pd.read_csv(r"MasterData\symptom_precaution.csv")


def query_llm(description, precautions, disease):
    prompt = (
        f"Patient has been diagnosed with **{disease}**.\n"
        f"Description: {description}\n"
        f"Precautions: {', '.join(precautions)}\n"
        f"Provide a user-friendly, supportive response as if you are a helpful doctor."
    )

    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 250,
            "temperature": 0.7,
            "do_sample": True
        }
    }

    response = requests.post(
        f"https://api-inference.huggingface.co/models/{MODEL}",
        headers=headers,
        json=data
    )

    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"Error from model: {response.status_code}, {response.text}"
    
    
def get_disease_info(predicted_disease):
    # Find the description
    description_row = description_df[description_df.iloc[:, 0].str.lower() == predicted_disease.lower()]
    description = description_row.iloc[0, 1] if not description_row.empty else "No description available."

    # Find the precautions (all columns after the first)
    precaution_row = precaution_df[precaution_df.iloc[:, 0].str.lower() == predicted_disease.lower()]
    precautions = precaution_row.iloc[0, 1:].dropna().tolist() if not precaution_row.empty else ["No precautions available."]

    return description, precautions

def extract_symptoms(user_input):
    valid_symptoms = get_valid_symptoms()
    input_lower = user_input.lower()
    found = [symptom for symptom in valid_symptoms if symptom.replace('_', ' ') in input_lower or symptom in input_lower]
    return found

if __name__ == "__main__":
    user_input = input("Enter your symptoms: ")
    extracted = extract_symptoms(user_input)

    if not extracted:
        print("Sorry, I couldn't recognize any valid symptoms from your input.")
    else:
        disease = predict_disease(extracted)
        desc,precs = get_disease_info(disease)
        # print(f"Predicted Disease: {disease}")
        # print("Description:",desc)
        # print("Precautions:",end=" ")
        # for i in precs:
        #     print(i,end=',')
        print("Response :",query_llm(desc,precs,disease))
        
