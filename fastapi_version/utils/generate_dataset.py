import pandas as pd
from generate_features import extract_features_from_url
from tqdm import tqdm

#Charging the CSV 
df = pd.read_csv("../data/malicious_phish1.csv")

# List to stock the extracted features
all_features = []

# Extracting the features of each URL
for i, row in tqdm(df.iterrows(), total=len(df)):
    url = row["url"]
    label = row["type"]  # 0 or 1
    features = extract_features_from_url(url)
    features["url"] = url
    features["label"] = label
    all_features.append(features)

# Create a new dataframe
df_new = pd.DataFrame(all_features)

#  Save it in a new CSV
df_new.to_csv("../data/features_dataset.csv", index=False)
print("New file names : features_dataset.csv is generated !")
