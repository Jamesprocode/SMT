import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import sys
sys.path.append('./SMT-plusplus')

from smt_model import SMTModelForCausalLM
from data_augmentation.data_augmentation import convert_img_to_tensor
from eval_functions import compute_poliphony_metrics

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

# Load model
model_id = "PRAIG/smt-fp-grandstaff"
print(f"Loading model: {model_id}")
model = SMTModelForCausalLM.from_pretrained(model_id).to(device)
model.eval()
print("Model loaded!\n")

# Load dataset
dataset = load_dataset("PRAIG/grandstaff")
test_set = dataset['test']
print(f"Test set size: {len(test_set)}\n")

predictions = []
ground_truths = []
failed = 0

print("Starting evaluation...")
for idx in tqdm(range(min(10,len(test_set)))): # Change 10 to len(test_set) for full evaluation
    try:
        sample = test_set[idx]
        image = sample['image']  # PIL Image
        
        # Convert to tensor (no resize)
        x = convert_img_to_tensor(np.array(image)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_sequence, _ = model.predict(x, convert_to_str=True)
        
        pred_str = "".join(pred_sequence).replace('<b>', '\n').replace('<s>', ' ').replace('<t>', '\t')
        
        # --- THIS IS THE FIX ---
        # Get the ground truth and convert it from EKERN to raw KERN to match the model's output
        gt_str = sample['transcription']
        gt_str = gt_str.replace('**ekern_1.0', '**kern')
        gt_str = gt_str.replace('·', '')
        # --- END OF FIX ---

        
        predictions.append(pred_str)
        ground_truths.append(gt_str)
        
    except Exception as e:
        print(f"\nError on sample {idx}: {e}")
        failed += 1

# Compute metrics
print(f"\nSuccessfully processed: {len(predictions)}/{len(test_set)}")
print(f"Failed: {failed}\n")

if len(predictions) > 0:
    cer, ser, ler = compute_poliphony_metrics(predictions, ground_truths)
    
    print("="*50)
    print("RESULTS")
    print("="*50)
    print(f"CER: {cer:.2f}%")
    print(f"SER: {ser:.2f}%")
    print(f"LER: {ler:.2f}%")
    
    # Show a sample prediction vs ground truth
    print("\n" + "="*50)
    print("Sample Prediction (first 200 chars):")
    print("="*50)
    print("Prediction:", predictions[0][:200])
    print("\nGround Truth:", ground_truths[0][:200])
else:
    print("No successful predictions")
    
    
    # python evaluateion_kern.py --dataset_name "PRAIG/camera-grandstaff" --num_samples 100
    
    
    #not name myself just name the project
    #publication not in the top distinguish small project and big project
    # opening paragraph about myself especcially for career fair
    
    
#     CER: 945.11%
# SER: 1199.04%
# LER: 1328.12%

