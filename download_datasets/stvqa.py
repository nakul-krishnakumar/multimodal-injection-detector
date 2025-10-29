"""
Download a sample of 2000 entries from the ST-VQA dataset
Memory-efficient batch processing to avoid system freezes
"""

import json
from pathlib import Path

from datasets import load_dataset

# Configuration
TOTAL_SAMPLES = 2000
BATCH_SIZE = 100  # Process 100 entries at a time to avoid memory issues

# Create output directory
output_dir = Path("data/ST-VQA")
output_dir.mkdir(parents=True, exist_ok=True)
images_dir = output_dir / "images"
images_dir.mkdir(exist_ok=True)

print(
    f"Downloading {TOTAL_SAMPLES} entries from ST-VQA dataset in batches of {BATCH_SIZE}..."
)
print("Using streaming mode to minimize memory usage.")

# Load dataset in streaming mode - using train split
dataset = load_dataset("vikhyatk/st-vqa", split="train", streaming=True)

# Initialize counters and storage
all_metadata = []
first_two_entries = []
total_processed = 0

# Process in batches
batch = []
for i, item in enumerate(dataset):
    if i >= TOTAL_SAMPLES:
        break

    batch.append(item)

    # When batch is full or we've reached the end, process it
    if len(batch) == BATCH_SIZE or i == TOTAL_SAMPLES - 1:
        print(
            f"Processing batch: entries {total_processed + 1} to {total_processed + len(batch)}..."
        )

        # Process each item in the batch
        for idx, item in enumerate(batch):
            entry_id = total_processed + idx

            # Create image filename
            image_filename = f"stvqa_{entry_id:06d}.jpg"

            # Extract question-answer pairs from 'qas' field
            qa_pairs = item.get("qas", [])

            # Save metadata
            entry = {
                "entry_id": entry_id,
                "image_filename": image_filename,
                "question_answer_pairs": qa_pairs,
                "num_qa_pairs": len(qa_pairs) if isinstance(qa_pairs, list) else 0,
                "has_image": "image" in item and item.get("image") is not None,
            }
            all_metadata.append(entry)

            # Save first 2 entries for preview
            if len(first_two_entries) < 2:
                preview_entry = entry.copy()
                first_two_entries.append(preview_entry)

            # Save image immediately to disk (don't keep in memory)
            if "image" in item and item.get("image") is not None:
                image_path = images_dir / image_filename
                try:
                    image_obj = item.get("image")
                    if hasattr(image_obj, "save"):
                        image_obj.save(image_path)
                except Exception as e:
                    print(f"Warning: Failed to save image {image_filename}: {e}")

        total_processed += len(batch)
        print(f"✓ Processed {total_processed}/{TOTAL_SAMPLES} entries")

        # Clear batch to free memory
        batch = []

print("=" * 80)
print("First 2 entries:")
print("=" * 80)
for i, entry in enumerate(first_two_entries):
    print(f"\nEntry {i + 1}:")
    print(json.dumps(entry, indent=2, default=str))

# Save all metadata to JSON
output_file = output_dir / "stvqa_2000_samples.json"
print("=" * 80)
print(f"Saving metadata to {output_file}...")

with open(output_file, "w") as f:
    json.dump(all_metadata, f, indent=2)

print(f"✓ Saved {len(all_metadata)} entries to {output_file}")
print(f"✓ Saved images to {images_dir}")
print("\nDone! Memory-efficient batch processing completed successfully.")
