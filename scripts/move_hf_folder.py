import os
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
from dotenv import load_dotenv

load_dotenv()
api = HfApi()

REPO_ID = "naaakuuul/prompt-injection-classifier"
OLD_FOLDER = "DocVQA_Injection"
NEW_FOLDER = "new_docvqa_folder"  # Example new folder name

try:
    # First, list all files in the directory to be moved
    repo_files = api.list_repo_files(repo_id=REPO_ID)

    operations = []
    for file_info in repo_files:
        original_path = file_info.path

        # Determine the new path for each file
        relative_path = os.path.relpath(original_path, start=OLD_FOLDER)
        new_path = os.path.join(NEW_FOLDER, relative_path).replace("\\", "/")

        # Add operations to copy and delete the files
        operations.append(
            CommitOperationAdd(path_in_repo=new_path, path_or_fileobj=original_path)
        )
        operations.append(CommitOperationDelete(path_in_repo=original_path))

    # Check if there are any files to move before committing
    if operations:
        api.create_commit(
            repo_id=REPO_ID,
            operations=operations,
            commit_message=f"Move folder '{OLD_FOLDER}' to '{NEW_FOLDER}'",
        )
        print("Successfully renamed folder.")
    else:
        print(f"No files found in '{OLD_FOLDER}'. Nothing to rename.")

except Exception as e:
    print(f"Error renaming folder: {e}")
