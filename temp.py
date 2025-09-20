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

model = SMTModelForCausalLM.from_pretrained("antoniorv6/smt-grandstaff").to(device)



x = convert_img_to_tensor(np.array(image)).unsqueeze(0).to(device)

predictions, _ = model.predict(x, convert_to_str=True)

print("".join(predictions).replace('<b>', '\n').replace('<s>', ' ').replace('<t>', '\t'))
# replace <model_reference> with the model you want to use, e.g. "SMT/smt-next-grandstaff"

