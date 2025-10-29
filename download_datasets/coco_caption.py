"""
Download a sample of 2000 entries from the COCO-Caption2017 dataset
Memory-efficient batch processing to avoid system freezes
"""
import json
from pathlib import Path

from datasets import load_dataset

# Configuration
TOTAL_SAMPLES = 2000
BATCH_SIZE = 100  # Process 100 entries at a time to avoid memory issues

# Create output directory
output_dir = Path("data/COCO-Caption2017")
output_dir.mkdir(parents=True, exist_ok=True)
images_dir = output_dir / "images"
images_dir.mkdir(exist_ok=True)

print(f"Downloading {TOTAL_SAMPLES} entries from COCO-Caption2017 dataset in batches of {BATCH_SIZE}...")
print("Using streaming mode to minimize memory usage.")

# Load dataset in streaming mode
dataset = load_dataset(
    "lmms-lab/COCO-Caption2017",
    split="val",
    streaming=True
)

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
    
    # print(f"Item {i}:", item)

    # When batch is full or we've reached the end, process it
    if len(batch) == BATCH_SIZE or i == TOTAL_SAMPLES - 1:
        print(f"Processing batch: entries {total_processed + 1} to {total_processed + len(batch)}...")
        
        # Process each item in the batch
        for idx, item in enumerate(batch):
            entry_id = total_processed + idx
            
            # Extract image filename if available, otherwise create one
            image_filename = None
            if "image" in item and item.get("image") is not None:
                # The image is a PIL Image object, create a filename
                image_filename = f"coco_{entry_id:06d}.jpg"
            
            # COCO-Caption2017 structure:
            # - 'question': The prompt (e.g., "Please carefully observe the image and come up with a caption for the image.")
            # - 'answer': List of 5 captions
            # - 'question_id': Image filename (e.g., "000000293044.jpg")
            # - Other fields: id, license, file_name, coco_url, height, width, date_captured
            
            # Save metadata
            entry = {
                "entry_id": entry_id,
                "question_id": item.get("question_id"),
                "image_filename": image_filename,
                "question": item.get("question"),
                "captions": item.get("answer", []),  # List of captions
                "id": item.get("id"),
                "license": item.get("license"),
                "file_name": item.get("file_name"),
                "coco_url": item.get("coco_url"),
                "height": item.get("height"),
                "width": item.get("width"),
                "date_captured": item.get("date_captured"),
                "has_image": image_filename is not None
            }
            all_metadata.append(entry)
            
            # Save first 2 entries for preview
            if len(first_two_entries) < 2:
                preview_entry = entry.copy()
                first_two_entries.append(preview_entry)
            
            # Save image immediately to disk (don't keep in memory)
            if image_filename:
                image_path = images_dir / image_filename
                try:
                    image_obj = item.get("image")
                    if hasattr(image_obj, 'save'):
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
output_file = output_dir / "coco_caption_2000_samples.json"
print("=" * 80)
print(f"Saving metadata to {output_file}...")

with open(output_file, 'w') as f:
    json.dump(all_metadata, f, indent=2)

print(f"✓ Saved {len(all_metadata)} entries to {output_file}")
print(f"✓ Saved images to {images_dir}")
print("\nDone! Memory-efficient batch processing completed successfully.")
