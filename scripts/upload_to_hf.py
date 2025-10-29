from huggingface_hub import HfApi
from dotenv import load_dotenv 

load_dotenv()

api = HfApi()
api.upload_large_folder(
    folder_path="./data/injected",
    repo_id="naaakuuul/prompt-injection-classifier",
    repo_type="dataset",
)
