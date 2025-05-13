import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Load and prepare data
df = pd.read_csv(r"Data\dataset.csv", header=None)
df = df.drop_duplicates().reset_index(drop=True)
disease_col = df.columns[0]
symptom_cols = df.columns[1]

# Create master symptom list
# all_symptoms = sorted(set(sym for row in df[symptom_cols].values for sym in row if pd.notna(sym)))


all_symptoms = sorted(set(
    sym
    for row in df[symptom_cols].values
    if isinstance(row, (list, tuple)) or hasattr(row, '__iter__')
    for sym in (row if isinstance(row, (list, tuple)) else [row])
    if pd.notna(sym)
))


X = []
y=df.iloc[:,0]
for idx in range(len(df)):
    symptoms_present = set([sym for sym in df.iloc[idx] if pd.notna(sym)])
    # print((symptoms_present))
    binary_row = [1 if symptom in symptoms_present else 0 for symptom in all_symptoms]
    X.append(binary_row)
    

# Remove duplicates
df_encoded = pd.DataFrame(X, columns=all_symptoms)
df_encoded["Disease"] = y


# Final inputs and labels
X_clean = df_encoded[all_symptoms]
y_clean = df_encoded["Disease"]

# Label encode diseases
le = LabelEncoder()
y_encoded = le.fit_transform(y_clean)

# Train the classifier
clf = DecisionTreeClassifier()
clf.fit(X_clean, y_encoded)

# Prediction interface
def predict_disease(symptoms_list):
    input_vector = [1 if symptom in symptoms_list else 0 for symptom in all_symptoms]
    pred = clf.predict([input_vector])[0]
    return le.inverse_transform([pred])[0]

# Export valid symptoms
def get_valid_symptoms():
    return all_symptoms
