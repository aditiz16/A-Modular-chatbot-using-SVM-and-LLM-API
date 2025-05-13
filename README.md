# A modular chatbot using SVM and LLM API
A Modular Toolkit for Symptom-Based Disease Classification, Visualization, and Synthetic Data Generation
This is documentation of everything I learned through while building this project. 

Steps to Follow to Understand this project : 

- My Idea behind this project
- Choosing the dataset
- Visualising the dataset
- Architecture of the project
- Implementation
 
---
## My Idea behind this project 

The core idea was to create a modular toolkit that could do more than just classify diseases based on symptoms . I wanted it to 
- Predict disease based on the symptom using ML classifiers
- Visualise patterns in symptom to improve understanding of the dataset.
- Integrate with LLMs to allow a conversational interface for medical queris 

This was both a learning exercise and a prototype for how ML + LLMs can assist in real-world health applications.
---
## Choosing the Dataset
I used publicly available structured datasets related to healthcare symptoms and diseases. These included:
- `dataset.csv` - Includes the disease and symptoms related to the disease.
- `symptom_severity.csv` – Maps symptoms to severity scores.
- `symptom_precaution.csv` – Suggests precautions for each disease.
- `symptom_description.csv` – Short text-based descriptions of symptoms.

The data was pre-cleaned but still needed formatting, label encoding, and transformation for use with machine learning models.
---
## Visualising the dataset 

Before jumping into ML, I visualised the dataset using: 
Before jumping into ML, I visualized the dataset using:
- **Matplotlib** and **Seaborn** for heatmaps, bar graphs, and symptom frequency plots.
- Explored co-occurrence of symptoms.
- Severity histograms to understand class imbalance.

This helped me identify:
- Most common symptoms
- Rare diseases (which might need synthetic data generation ~ not implemented in this maybe in future I will add the same)
- Dataset biases

---

## Architecture of this project 

Here's how the system is architected: 

![image](https://github.com/user-attachments/assets/215ea83f-e5ab-41ff-b73a-ea50b99e701f)

---

## Implementation Overview : 

- `main.py`: Orchestrates the interaction with the user and integration of API
- `visualize.py`: Contains dataset visualizations
- `classifier.py`: Trains an SVM on symptom-disease data.

The backend is in **Python**, with **scikit-learn** for ML, **pandas** for data handling, and **requests** for API interaction.
---
## Understanding Classifiers 

Before building the model, I studied the basics of ML classifiers:

- **Decision Trees**: Intuitive models that split data based on feature importance. Easy to visualize and explain but prone to overfitting.
- **Support Vector Machines (SVM)**: High-performance classifiers that work well for high-dimensional data. Good at handling binary and multi-class classification with clear margins.

I tested both and chose SVM for its better accuracy on the cleaned dataset.

---
This project helped me combine : 
- Classical ML
- Data engineering
- LLM integration with APIs
---
I will continue to improve this project over time. Feel free to explore, fork and build upon it. 
---
Disclaimer: The exact original source of the datasets (dataset.csv, symptom_severity.csv, symptom_precaution.csv, and symptom_description.csv) is currently unknown. These files have been circulated in public repositories and educational projects related to symptom-based disease classification, particularly on platforms like Kaggle and GitHub.

If you are the original author or know the official source, please let me know so I can properly attribute and cite the dataset.
---



