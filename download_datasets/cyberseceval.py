import os
import torch
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from PIL import Image
import pandas as pd
# from tqdm.auto import tqdm

load_dotenv()

DATA_DIR = "./data/cyberseceval3-visual-prompt-injection"
IMAGES_DIR = os.path.join(DATA_DIR, "images")

def attach_image(row):
    image_path = os.path.join(IMAGES_DIR, f"{row['id']}.png")
    row["image_path"] = image_path
    try:
        row["image"] = Image.open(image_path).convert("RGB")
    except:
        row["image"] = None
    
    return row

def download():
    device = "cpu"
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        device = "cuda"

    local_dir = snapshot_download(
        repo_id="facebook/cyberseceval3-visual-prompt-injection", 
        repo_type="dataset", 
        local_dir=DATA_DIR,
        token=os.getenv("HF_TOKEN")
    )

def preprocess_dataset():
    test_cases_path = os.path.join(DATA_DIR, "test_cases.json")

    dataset = pd.read_json(test_cases_path)
    dataset = dataset.apply(attach_image, axis=1)

    output_path = os.path.join(DATA_DIR, "test_cases_with_images.pkl")
    dataset.to_pickle(output_path)
    print(f"Dataset saved to: {output_path}")

def download__cyberseceval3():
    download()

    preprocess_dataset()


if __name__ == "__main__":
    download__cyberseceval3()
