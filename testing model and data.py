import torch
from transformers import AutoModel, AutoConfig
from datasets import load_dataset

from smt_model import SMTModelForCausalLM

print("=" * 50)
print("Testing Model Loading")
print("=" * 50)

# Test 1: Load pre-trained model directly from HuggingFace
try:
    model_id = "PRAIG/smt-fp-grandstaff"  # or PRAIG/sheet-music-transformer
    print(f"Loading model from HuggingFace: {model_id}...")
    model = SMTModelForCausalLM.from_pretrained(model_id)
    print("✓ Model loaded successfully!")
    print(f"  Model type: {type(model)}")
    # print(f"  Config: {model.config}")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    print("  Try checking the exact model name on HuggingFace")

print("\n" + "=" * 50)
print("Testing Dataset Loading")
print("=" * 50)

# Test 2: Load datasets directly from HuggingFace
datasets_to_test = {
    "grandstaff": "PRAIG/grandstaff",
    "camera-grandstaff": "PRAIG/camera-grandstaff",
    "fp-grandstaff": "PRAIG/fp-grandstaff",
    "grandstaff-ekern": "PRAIG/grandstaff-ekern",
}

for name, repo_id in datasets_to_test.items():
    try:
        print(f"\nLoading {name} from {repo_id}...")
        dataset = load_dataset(repo_id)
        print(f"✓ {name} loaded successfully!")
        print(f"  Splits: {list(dataset.keys())}")
        if 'train' in dataset:
            print(f"  Train size: {len(dataset['train'])}")
        if 'test' in dataset:
            print(f"  Test size: {len(dataset['test'])}")
    except Exception as e:
        print(f"✗ {name} loading failed: {e}")

print("\n" + "=" * 50)
print("GPU Check")
print("=" * 50)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")