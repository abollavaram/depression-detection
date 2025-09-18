import pandas as pd

# Load the dataset
df = pd.read_csv("depression_clean.csv")

# Show top 5 rows
print("First 5 rows of the dataset:\n")
print(df.head())

# Show shape
print("\nDataset shape:", df.shape)

# Check for missing values
print("\nMissing values per column:\n")
print(df.isnull().sum())

# Check class distribution
print("\nClass distribution (0 = not depressed, 1 = depressed):")
print(df['label'].value_counts())
