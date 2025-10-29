"""
Fix ChartQA metadata by manually loading the dataset structure
This avoids the numpy/pyarrow dependency issues
"""
import json
from pathlib import Path

print("Fixing ChartQA metadata...")

# The issue is that the metadata has null values
# Let's check what we have
chartqa_json = Path("data/ChartQA/chartqa_2000_samples.json")

if not chartqa_json.exists():
    print(f"Error: {chartqa_json} not found")
    exit(1)

with open(chartqa_json, 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data)} entries")
print(f"\nFirst entry structure:")
print(json.dumps(data[0], indent=2))

# Count null entries
null_count = sum(1 for e in data if e.get('question') is None)
print(f"\nEntries with null question: {null_count}/{len(data)}")

if null_count > 0:
    print("\n⚠️  The dataset was not downloaded correctly.")
    print("The ChartQA dataset structure might be different than expected.")
    print("\nTo fix this, you need to:")
    print("1. Check the actual ChartQA dataset structure on HuggingFace")
    print("2. Update the download_chartqa.py script with the correct field names")
    print("3. Re-run the download script")
    print("\nAlternatively, delete the data/ChartQA directory and re-download:")
    print("  rm -rf data/ChartQA")
    print("  python3 download_chartqa.py")
else:
    print("\n✓ Metadata looks good!")
