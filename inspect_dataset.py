from datasets import load_dataset

try:
    ds = load_dataset("MathArena/apex_2025", split="train")
    print("Dataset columns:", ds.column_names)
    print("First example:", ds[0])
except Exception as e:
    print(f"Error loading dataset: {e}")
