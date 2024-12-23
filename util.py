import torch

pt = torch.load("/scratch/vidit_a_mfs.iitr/finetune-and-attack/probs/finetune/quantized/eval/0.pt")

print(pt.shape)