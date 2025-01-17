import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("reviews_formatted.csv")

df["label"] = df["label"].apply(lambda x: "Pozitiv" if x == 1 else "Negativ")

train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

train_df.to_csv("train_dataset.csv", index=False)
val_df.to_csv("val_dataset.csv", index=False)
test_df.to_csv("test_dataset.csv", index=False)

print("Seturi salvate:")
print(f"Training set: {len(train_df)} exemple")
print(f"Validation set: {len(val_df)} exemple")
print(f"Test set: {len(test_df)} exemple")