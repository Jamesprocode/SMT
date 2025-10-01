import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import sys
import argparse
import re

# Make sure this path is correct for your setup
# sys.path.append('./SMT-plusplus') 

from smt_model import SMTModelForCausalLM
from data_augmentation.data_augmentation import convert_img_to_tensor
from eval_functions import compute_poliphony_metrics

def convert_to_bekern(raw_kern_string):
    """
    Converts a raw KERN string to a BEKERN-like format by splitting symbols.
    This is based on the paper's description of BEKERN as a semantic split.
    Example: '8cc-/L' -> '8 cc / L'
    """
    def split_token(token):
        if token in ['*clefF4', '*clefG2', '*M2/4', '**kern', '=', '*-', '.'] or token.startswith('*k['):
            return token
        # Regex to split duration, accidentals, pitch, octave, and articulations/slurs
        parts = re.findall(r'(\d+\.?)?([#n\-]*)?([a-gA-G]+)(\.*)?([\/\\\[\]_]*)?', token)
        if parts:
            return ' '.join(filter(None, parts[0]))
        return token

    bekern_lines = []
    for line in raw_kern_string.split('\n'):
        if line.startswith('*') or line in ['=', '*-']:
            bekern_lines.append(line)
            continue
        
        columns = line.split('\t')
        new_columns = []
        for col in columns:
            chords = col.split(' ')
            new_chords = [' '.join(split_token(note) for note in chord.split('.')) for chord in chords]
            new_columns.append(' '.join(new_chords))
        bekern_lines.append('\t'.join(new_columns))
        
    return '\n'.join(bekern_lines)


def main(dataset_name, num_samples):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Load model
    model_id = "PRAIG/smt-fp-grandstaff"
    print(f"Loading model: {model_id}")
    model = SMTModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()
    print("Model loaded!\n")

    # Load specified dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    test_set = dataset['test']
    print(f"Test set size: {len(test_set)}\n")

    predictions = []
    ground_truths = []
    failed = 0

    num_samples_to_test = len(test_set) if num_samples == -1 else min(num_samples, len(test_set))
    
    print(f"Starting evaluation on {num_samples_to_test} samples...")

    for idx in tqdm(range(num_samples_to_test)):
        try:
            sample = test_set[idx]
            image = sample['image']
            
            x = convert_img_to_tensor(np.array(image)).unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred_sequence, _ = model.predict(x, convert_to_str=True)
            
            # 1. Get model's raw KERN output
            raw_pred_str = "".join(pred_sequence).replace('<b>', '\n').replace('<s>', ' ').replace('<t>', '\t')
            
            # --- THIS IS THE FIX ---
            # 2. Convert the raw KERN prediction to BEKERN
            bekern_pred_str = convert_to_bekern(raw_pred_str)
            
            # 3. Get the ground truth (which is already in BEKERN format)
            gt_str = sample['transcription']
            
            predictions.append(bekern_pred_str)
            ground_truths.append(gt_str)
            
        except Exception as e:
            print(f"\nError on sample {idx}: {e}")
            failed += 1

    print(f"\nSuccessfully processed: {len(predictions)}/{num_samples_to_test}")
    print(f"Failed: {failed}\n")

    if len(predictions) > 0:
        # compute_poliphony_metrics expects strings, it will calculate edit distance on the BEKERN tokens.
        cer, ser, ler = compute_poliphony_metrics(predictions, ground_truths)
        
        print("="*50)
        print(f"RESULTS for {dataset_name} (BEKERN Comparison)")
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
        
            # Show a sample prediction vs ground truth
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained SMT model on a given dataset.")
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset from Hugging Face.')
    # --- THIS ARGUMENT IS UPDATED ---
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=10, 
        help='Number of samples to evaluate. Default is 10. Set to -1 to evaluate all.'
    )
    # --- END OF UPDATE ---
    args = parser.parse_args()
    main(args.dataset_name, args.num_samples)