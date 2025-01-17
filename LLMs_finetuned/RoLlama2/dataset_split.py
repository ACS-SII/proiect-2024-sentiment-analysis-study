import pandas as pd
from sklearn.model_selection import train_test_split

# Citește dataset-ul inițial
df = pd.read_csv("reviews_formatted.csv")

# Convertim etichetele binare la "Pozitiv" și "Negativ"
df["label"] = df["label"].apply(lambda x: "Pozitiv" if x == 1 else "Negativ")

# Împărțim dataset-ul în Training (80%) și Test+Validation (20%), păstrând proporțiile claselor
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# Împărțim restul pentru Validation (10%) și Test (10%)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

# Salvăm seturile în fișiere CSV separate
train_df.to_csv("train_dataset.csv", index=False)
val_df.to_csv("val_dataset.csv", index=False)
test_df.to_csv("test_dataset.csv", index=False)

print("Seturi salvate:")
print(f"Training set: {len(train_df)} exemple")
print(f"Validation set: {len(val_df)} exemple")
print(f"Test set: {len(test_df)} exemple")