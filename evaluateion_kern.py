import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import sys
import argparse

# Make sure this path is correct for your setup
# sys.path.append('./SMT-plusplus') 

from smt_model import SMTModelForCausalLM
from data_augmentation.data_augmentation import convert_img_to_tensor
from eval_functions import compute_poliphony_metrics

def normalize_ground_truth(gt_str):
    """
    Normalizes any ground truth format (EKERN, BEKERN) to a common,
    space-delimited format to match the model's BEKERN-like output.
    """
    # Remove header declarations
    gt_str = gt_str.replace('**ekern_1.0', '**kern').replace('**bekern_1.0', '**kern')
    
    # Replace EKERN's dot separator with a space
    gt_str = gt_str.replace('·', '')
    
    return gt_str

def main(dataset_name, num_samples):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    model_id = "PRAIG/smt-fp-grandstaff"
    print(f"Loading model: {model_id}")
    model = SMTModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()
    print("Model loaded!\n")

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    test_set = dataset['test']
    print(f"Test set size: {len(test_set)}\n")

    predictions = []
    ground_truths = []
    failed = 0
    
    if num_samples == -1:
        num_samples_to_test = len(test_set)
    else:
        num_samples_to_test = min(num_samples, len(test_set))
    
    print(f"Starting evaluation on {num_samples_to_test} samples...")

    for idx in tqdm(range(num_samples_to_test)):
        try:
            sample = test_set[idx]
            image = sample['image']
            x = convert_img_to_tensor(np.array(image)).unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred_sequence, _ = model.predict(x, convert_to_str=True)
            
            # 1. Get model's raw output and convert to BEKERN-like format (with spaces)
            pred_str_raw = "".join(pred_sequence).replace('<b>', '\n').replace('<s>', ' ').replace('<t>', '\t')
            # Assuming the model's internal logic already separates tokens semantically
            
            # 2. Get the ground truth and normalize it to the same space-delimited format
            gt_str_raw = sample['transcription']
            gt_str_normalized = normalize_ground_truth(gt_str_raw)

            predictions.append(pred_str_raw)
            ground_truths.append(gt_str_normalized)
            
        except Exception as e:
            print(f"\nError on sample {idx}: {e}")
            failed += 1

    print(f"\nSuccessfully processed: {len(predictions)}/{num_samples_to_test}")
    print(f"Failed: {failed}\n")

    if len(predictions) > 0:
        cer, ser, ler = compute_poliphony_metrics(predictions, ground_truths)
        print("="*50)
        print(f"RESULTS for {dataset_name} (Normalized Comparison)")
        print("="*50)
        print(f"CER: {cer:.2f}%")
        print(f"SER: {ser:.2f}%")
        print(f"LER: {ler:.2f}%")
        
        print("\n" + "="*50)
        print("Sample Prediction (first 200 chars):")
        print("="*50)
        print("Prediction:", predictions[0][:200])
        print("\nGround Truth:", ground_truths[0][:200])
    else:
        print("No successful predictions")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained SMT model on a given dataset.")
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset from Hugging Face.')
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=10, 
        help='Number of samples to evaluate. Default is 10. Set to -1 to evaluate all.'
    )
    args = parser.parse_args()
    main(args.dataset_name, args.num_samples)
