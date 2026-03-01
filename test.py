import pandas as pd

# Read the parquet files
train_df = pd.read_parquet('data/train.parquet')
test_df = pd.read_parquet('data/test.parquet')
val_df = pd.read_parquet('data/validation.parquet')

# Display information about each dataset
print("=" * 50)
print("TRAIN DATA")
print("=" * 50)
print(f"Shape: {train_df.shape}")
print(f"\nColumns: {train_df.columns.tolist()}")
print(f"\nData Types:\n{train_df.dtypes}")
print(f"\nFirst few rows:\n{train_df.head()}")
print(f"\nMissing values:\n{train_df.isnull().sum()}")

print("\n" + "=" * 50)
print("TEST DATA")
print("=" * 50)
print(f"Shape: {test_df.shape}")
print(f"\nColumns: {test_df.columns.tolist()}")
print(f"\nData Types:\n{test_df.dtypes}")
print(f"\nFirst few rows:\n{test_df.head()}")
print(f"\nMissing values:\n{test_df.isnull().sum()}")

print("\n" + "=" * 50)
print("VALIDATION DATA")
print("=" * 50)
print(f"Shape: {val_df.shape}")
print(f"\nColumns: {val_df.columns.tolist()}")
print(f"\nData Types:\n{val_df.dtypes}")
print(f"\nFirst few rows:\n{val_df.head()}")
print(f"\nMissing values:\n{val_df.isnull().sum()}")