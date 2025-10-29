"""
Visualize sample entries from the processed dataset
Shows original vs injected images side by side
"""
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configuration
DATASET_PATH = "data/DocVQA_Injection/docvqa_injection_dataset.json"
ORIGINAL_IMAGES_DIR = "data/DocVQA/images"
NUM_SAMPLES = 5

def visualize_samples():
    """Visualize sample entries with original and injected images"""
    
    # Load dataset
    print(f"Loading dataset from {DATASET_PATH}...")
    if not Path(DATASET_PATH).exists():
        print(f"Error: Dataset not found at {DATASET_PATH}")
        print("Please run preprocess_docvqa_to_injection.py first")
        return
    
    with open(DATASET_PATH, 'r') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} entries")
    
    # Select samples
    num_to_show = min(NUM_SAMPLES, len(dataset))
    samples = dataset[:num_to_show]
    
    print(f"\nVisualizing {num_to_show} samples...")
    
    for idx, entry in enumerate(samples):
        print(f"\n{'='*80}")
        print(f"SAMPLE {idx + 1}")
        print(f"{'='*80}")
        
        # Print metadata
        print(f"ID: {entry['id']}")
        print(f"User Input: {entry['user_input_text']}")
        print(f"Image Description: {entry['image_description']}")
        print(f"Image Text (Injection): {entry['image_text'][:100]}...")
        print(f"Injection Technique: {entry['injection_technique']}")
        print(f"Injection Type: {entry['injection_type']}")
        print(f"Risk Category: {entry['risk_category']}")
        print(f"Judge Question: {entry['judge_question']}")
        print(f"\nSystem Prompt: {entry['system_prompt'][:100]}...")
        
        # Load images
        # original_image_path = Path(ORIGINAL_IMAGES_DIR) / f"{entry['original_question_id']}.png"
        injected_image_path = Path("data/DocVQA_Injection") / entry['image_path']
        
        # if not original_image_path.exists():
        #     print(f"Warning: Original image not found: {original_image_path}")
        #     continue
        
        if not injected_image_path.exists():
            print(f"Warning: Injected image not found: {injected_image_path}")
            continue
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f"Sample {idx + 1}: {entry['user_input_text'][:60]}...", 
                     fontsize=14, fontweight='bold')
        
        # # Original image
        # original_img = Image.open(original_image_path)
        # axes[0].imshow(original_img)
        # axes[0].set_title("Original DocVQA Image", fontsize=12, fontweight='bold')
        # axes[0].axis('off')
        
        # Injected image
        injected_img = Image.open(injected_image_path)
        axes[1].imshow(injected_img)
        axes[1].set_title("Injected Image (with adversarial text)", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Add metadata box
        metadata_text = (
            f"Injection Type: {entry['injection_type']}\n"
            f"Risk: {entry['risk_category']}\n"
            f"Technique: {', '.join(entry['injection_technique'][:2])}"
        )
        
        # Color code by risk category
        if entry['risk_category'] == 'security-violating':
            box_color = 'red'
        else:
            box_color = 'orange'
        
        fig.text(0.5, 0.02, metadata_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.3))
        
        plt.tight_layout()
        
        # Save visualization
        output_path = Path("data/DocVQA_Injection") / f"visualization_sample_{idx + 1}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved visualization to: {output_path}")
        
        # Show plot
        plt.show()
        plt.close()
    
    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Visualized {num_to_show} samples")
    print(f"Saved to: data/DocVQA_Injection/visualization_sample_*.png")


def print_statistics():
    """Print dataset statistics"""
    
    if not Path(DATASET_PATH).exists():
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return
    
    with open(DATASET_PATH, 'r') as f:
        dataset = json.load(f)
    
    print(f"\n{'='*80}")
    print("DATASET STATISTICS")
    print(f"{'='*80}")
    
    print(f"\nTotal Entries: {len(dataset)}")
    
    # Risk categories
    security_count = sum(1 for e in dataset if e['risk_category'] == 'security-violating')
    logic_count = sum(1 for e in dataset if e['risk_category'] == 'logic-violating')
    
    print(f"\nRisk Categories:")
    print(f"  Security-violating: {security_count} ({security_count/len(dataset)*100:.1f}%)")
    print(f"  Logic-violating: {logic_count} ({logic_count/len(dataset)*100:.1f}%)")
    
    # Injection types
    direct_count = sum(1 for e in dataset if e['injection_type'] == 'direct')
    indirect_count = sum(1 for e in dataset if e['injection_type'] == 'indirect')
    
    print(f"\nInjection Types:")
    print(f"  Direct: {direct_count} ({direct_count/len(dataset)*100:.1f}%)")
    print(f"  Indirect: {indirect_count} ({indirect_count/len(dataset)*100:.1f}%)")
    
    # Injection techniques
    all_techniques = []
    for e in dataset:
        all_techniques.extend(e['injection_technique'])
    
    from collections import Counter
    technique_counts = Counter(all_techniques)
    
    print(f"\nInjection Techniques (top 5):")
    for technique, count in technique_counts.most_common(5):
        print(f"  {technique}: {count}")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--stats':
        print_statistics()
    else:
        try:
            import matplotlib
            visualize_samples()
        except ImportError:
            print("Error: matplotlib not installed")
            print("Install with: pip install matplotlib")
            print("\nShowing statistics instead:")
            print_statistics()
