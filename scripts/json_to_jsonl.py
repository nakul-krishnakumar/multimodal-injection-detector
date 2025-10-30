import os
import json

# Set your dataset directory path
dataset_dir = "./dataset_metadata/"

for filename in os.listdir(dataset_dir):
    if filename.endswith(".json"):
        json_path = os.path.join(dataset_dir, filename)
        jsonl_path = os.path.join(dataset_dir, filename.replace(".json", ".jsonl"))

        # Read JSON array
        with open(json_path, "r", encoding="utf-8") as f_json:
            records = json.load(f_json)  # expects a list

        # Write JSONL
        with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
            for record in records:
                f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Converted: {json_path} -> {jsonl_path}")
