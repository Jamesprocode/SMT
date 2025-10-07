import torch
import cv2
from data_augmentation.data_augmentation import convert_img_to_tensor
from smt_model import SMTModelForCausalLM
import numpy as np

import datasets

dataset = datasets.load_dataset('PRAIG/grandstaff')


# image = cv2.imread("sample.jpg")
image = dataset['test'][0]['image']
  
device = "cuda" if torch.cuda.is_available() else "cpu"

# print("smt_model path:", SMTModelForCausalLM.__module__)
# print("data_augmentation path:", convert_img_to_tensor.__module__)

model = SMTModelForCausalLM.from_pretrained("PRAIG/smt-fp-grandstaff").to(device)



x = convert_img_to_tensor(np.array(image)).unsqueeze(0).to(device)

predictions, _ = model.predict(x, convert_to_str=True)

# print("".join(predictions).replace('<b>', '\n').replace('<s>', ' ').replace('<t>', '\t'))
# replace <model_reference> with the model you want to use, e.g. "SMT/smt-next-grandstaff"

#ssh jwang3180@login-ice.pace.gatech.edu ssh to ICE

# activate conda environment
# source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
# conda activate magick-env
# WANDB_DISABLED=true  python fp-train-1.py --config_path config/GrandStaff/grandstaff.json

# 
# checking GPU and requesting GPU resources
# srun --mem=64G --gres=gpu:H200:1 --time=4:00:00 --pty bash
# srun --mem=64G --gres=gpu:H200:1 --time=4:00:00 --pty bash



dataset = datasets.load_dataset("PRAIG/fp-grandstaff")
sample = dataset['test'][0]
print("GT format:", repr(sample['transcription'][:200]))
print("\nContains <b>?", '<b>' in sample['transcription'])
print("Contains \\n?", '\n' in sample['transcription'])

# scp -i ~/.ssh/lambda-key.pem ubuntu@192.222.58.136:/lambda/nfs/SMTTraining/SMT/requirements.txt ~/Downloads/
