import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd 
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE

df=pd.read_csv("Data/dataset.csv")
df = df.drop_duplicates().reset_index(drop=True)
# print(df.iloc[:,0])
# sns.countplot(y=df.iloc[:,0],order=df.iloc[:,0].value_counts().index)
# plt.title("Disease Frequency Distribution")
# plt.xlabel("Count")
# plt.ylabel("Disease")
# plt.show()

# df["Symptom Count"] = df.iloc[:, 1:].notna().sum(axis=1)
# sns.histplot(df["Symptom Count"], bins=range(1, 20), kde=False)
# plt.title("Number of Symptoms per Record")
# plt.xlabel("Symptom Count")
# plt.ylabel("Number of Records")
# plt.show()


# all_symptom_entries = df.iloc[:, 1:].values.flatten()
# symptom_counts = Counter(sym for sym in all_symptom_entries if pd.notna(sym))

# symptom_df = pd.DataFrame(symptom_counts.items(), columns=["Symptom", "Count"]).sort_values("Count", ascending=False)

# plt.figure(figsize=(10, 8))
# sns.barplot(data=symptom_df.head(30), y="Symptom", x="Count")
# plt.title("Top 30 Most Common Symptoms")
# plt.show()

disease_col = df.columns[0]
symptom_cols = df.columns[1]

# Create list of all unique symptoms (flatten the symptom columns)
all_symptoms = sorted(set(row for row in df[symptom_cols].values ))
# print(all_symptoms)
# Create the binary feature matrix X
X = []
for idx in range(len(df)):
    symptoms_present = set([sym for sym in df.iloc[idx] if pd.notna(sym)])
    # print((symptoms_present))
    binary_row = [1 if symptom in symptoms_present else 0 for symptom in all_symptoms]
    X.append(binary_row)
    


binary_data = pd.DataFrame(X, columns=all_symptoms)
binary_data["Disease"] = df.iloc[:,0]

# Group by disease and average to get symptom presence ratio
symptom_disease_matrix = binary_data.groupby("Disease").mean()
# print(symptom_disease_matrix)
# plt.figure(figsize=(15, 10))
# sns.heatmap(symptom_disease_matrix, cmap="YlGnBu")
# plt.title("Symptom Presence Heatmap per Disease")
# plt.xlabel("Symptom")
# plt.ylabel("Disease")
# plt.show()


# corr_matrix = pd.DataFrame(X, columns=all_symptoms).corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, cmap="coolwarm", xticklabels=False, yticklabels=False)
# plt.title("Symptom Correlation Matrix")
# plt.show()

X_df = pd.DataFrame(X, columns=all_symptoms)
# y=df[disease_col]
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_df)

# # Plot
# plt.figure(figsize=(10, 7))
# sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='tab20', s=50, alpha=0.7)
# plt.title("PCA of Symptom Data")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.legend(title="Disease", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()

# t-SNE analysis 
label_encoder = LabelEncoder() 
y=label_encoder.fit_transform(df[disease_col])
disease_names=label_encoder.inverse_transform(sorted(set(y)))

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_df)

# Step 5: Plotting
tsne_df = pd.DataFrame(X_tsne, columns=["Dim 1", "Dim 2"])
tsne_df["Disease"] = label_encoder.inverse_transform(y)

plt.figure(figsize=(16, 10))
sns.scatterplot(
    data=tsne_df,
    x="Dim 1", y="Dim 2",
    hue="Disease",
    palette="tab20",
    legend="full",
    alpha=0.7
)
plt.title("t-SNE Visualization of Symptom Data")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.show()
