"""
Test script to verify preprocessing setup before running full pipeline
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

print("="*80)
print("PREPROCESSING SETUP TEST")
print("="*80)

# Test 1: Check environment variables
print("\n1. Checking Azure OpenAI environment variables...")
required_vars = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT_NAME"
]

all_present = True
for var in required_vars:
    value = os.getenv(var)
    if value:
        # Mask API key
        if "KEY" in var:
            display_value = value[:10] + "..." if len(value) > 10 else "***"
        else:
            display_value = value
        print(f"  ✓ {var}: {display_value}")
    else:
        print(f"  ✗ {var}: NOT SET")
        all_present = False

if not all_present:
    print("\n  ERROR: Some environment variables are missing!")
    print("  Please check your .env file")
else:
    print("\n  ✓ All environment variables are set")

# Test 2: Check DocVQA data
print("\n2. Checking DocVQA data...")
docvqa_json = "data/DocVQA/docvqa_3000_samples.json"
docvqa_images = "data/DocVQA/images"

if os.path.exists(docvqa_json):
    with open(docvqa_json, 'r') as f:
        data = json.load(f)
    print(f"  ✓ DocVQA JSON found: {len(data)} entries")
    
    # Check first entry
    if len(data) > 0:
        first_entry = data[0]
        print(f"  ✓ Sample entry keys: {list(first_entry.keys())}")
        
        # Check if image exists
        question_id = first_entry.get('questionId')
        image_path = Path(docvqa_images) / f"{question_id}.png"
        if image_path.exists():
            print(f"  ✓ Sample image found: {image_path.name}")
        else:
            print(f"  ✗ Sample image NOT found: {image_path}")
else:
    print(f"  ✗ DocVQA JSON NOT found: {docvqa_json}")
    print("  Please run: python3 download_docvqa.py")

if os.path.exists(docvqa_images):
    image_files = list(Path(docvqa_images).glob("*.png"))
    print(f"  ✓ DocVQA images directory found: {len(image_files)} images")
else:
    print(f"  ✗ DocVQA images directory NOT found: {docvqa_images}")

# Test 3: Check dependencies
print("\n3. Checking Python dependencies...")
dependencies = [
    ("openai", "Azure OpenAI client"),
    ("PIL", "Image processing"),
    ("tqdm", "Progress bars"),
    ("dotenv", "Environment variables")
]

for module_name, description in dependencies:
    try:
        if module_name == "PIL":
            import PIL
        elif module_name == "dotenv":
            import dotenv
        else:
            __import__(module_name)
        print(f"  ✓ {module_name}: {description}")
    except ImportError:
        print(f"  ✗ {module_name}: NOT INSTALLED ({description})")
        print(f"     Install with: pip install {module_name if module_name != 'PIL' else 'Pillow'}")

# Test 4: Check output directory
print("\n4. Checking output directory...")
output_dir = Path("data/DocVQA_Injection")
if output_dir.exists():
    print(f"  ✓ Output directory exists: {output_dir}")
else:
    print(f"  ℹ Output directory will be created: {output_dir}")

# Test 5: Test Azure OpenAI connection (optional)
print("\n5. Testing Azure OpenAI connection...")
try:
    from openai import AzureOpenAI
    
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
    
    if endpoint and api_key:
        client = AzureOpenAI(
            api_key=api_key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=endpoint
        )
        
        # Try a simple test call
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "Say 'test successful' if you can read this."}],
            max_tokens=10
        )
        
        print(f"  ✓ Azure OpenAI connection successful!")
        print(f"  ✓ Response: {response.choices[0].message.content}")
    else:
        print(f"  ⚠ Skipping connection test (credentials not set)")
        
except Exception as e:
    print(f"  ✗ Azure OpenAI connection failed: {e}")
    print(f"  Please check your credentials and endpoint")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if all_present and os.path.exists(docvqa_json):
    print("✓ Setup looks good! You can run the preprocessing pipeline:")
    print("  python3 preprocess_docvqa_to_injection.py")
else:
    print("✗ Setup incomplete. Please fix the issues above before running the pipeline.")

print("="*80)
