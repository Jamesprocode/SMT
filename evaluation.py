import os
# Disable git-lfs which triggers the SSL error
os.environ['HF_HUB_DISABLE_GIT_LFS_WARNING'] = '1'

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

# Load everything using download_mode="force_redownload" to bypass git
model_id = "PRAIG/smt-fp-grandstaff"
model = SMTModelForCausalLM.from_pretrained(model_id).to(device)
model.eval()

# Load dataset without git
dataset = load_dataset("PRAIG/fp-grandstaff", download_mode="reuse_cache_if_exists")
test_set = dataset['test']

predictions = []
ground_truths = []

print(f"Evaluating on {len(test_set)} samples...")

for idx in tqdm(range(min(10, len(test_set)))):  # Start with 10 samples to test
    sample = test_set[idx]
    image = np.array(sample['image'].convert('RGB'))
    
    import cv2
    width = int(np.ceil(image.shape[1] * 0.5))
    height = int(np.ceil(image.shape[0] * 0.5))
    image = cv2.resize(image, (width, height))
    
    x = convert_img_to_tensor(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_sequence, _ = model.predict(x, convert_to_str=True)
    
    pred_str = "".join(pred_sequence).replace('<b>', '\n').replace('<s>', ' ').replace('<t>', '\t')
    gt_str = sample['transcription']
    
    predictions.append(pred_str)
    ground_truths.append(gt_str)

cer, ser, ler = compute_poliphony_metrics(predictions, ground_truths)
print(f"\nResults (first 10 samples):")
print(f"CER: {cer:.2f}%")
print(f"SER: {ser:.2f}%")
print(f"LER: {ler:.2f}%")