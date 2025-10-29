"""
Test script to check ChartQA dataset structure
"""
from datasets import load_dataset
import json

print("Loading ChartQA dataset...")
dataset = load_dataset("HuggingFaceM4/ChartQA", split="train", streaming=True)

print("\nGetting first item...")
item = next(iter(dataset))

print("\n" + "="*80)
print("DATASET STRUCTURE")
print("="*80)
print(f"\nKeys: {list(item.keys())}")
print(f"\nTypes:")
for key, value in item.items():
    print(f"  {key}: {type(value)}")

print("\n" + "="*80)
print("SAMPLE VALUES")
print("="*80)
for key, value in item.items():
    if key == 'image':
        print(f"\n{key}: <PIL Image object>")
    else:
        value_str = str(value)
        if len(value_str) > 200:
            value_str = value_str[:200] + "..."
        print(f"\n{key}: {value_str}")

print("\n" + "="*80)
print("JSON REPRESENTATION (without image)")
print("="*80)
item_dict = {k: v for k, v in item.items() if k != 'image'}
print(json.dumps(item_dict, indent=2, default=str))
