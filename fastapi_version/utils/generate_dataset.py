import pandas as pd
from generate_features import extract_features_from_url
from tqdm import tqdm
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_file = os.path.join(base_dir, 'data', 'malicious_phish1.csv')
output_file = os.path.join(base_dir, 'data', 'features_dataset.csv')

os.makedirs(os.path.dirname(output_file), exist_ok=True)

df = pd.read_csv(input_file)
df.columns = df.columns.str.strip()

all_features = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    url = row["url"]
    label = row["type"]  # 'benign' or 'malicious'
    features = extract_features_from_url(url)
    features["url"] = url
    features["label"] = label
    all_features.append(features)

df_new = pd.DataFrame(all_features)
df_new.to_csv(output_file, index=False)

print("New file generated: features_dataset.csv")
