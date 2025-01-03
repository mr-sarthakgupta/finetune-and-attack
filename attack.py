import torch
import numpy as np
from torcheval.metrics import Perplexity
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from copy import deepcopy

def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"

    N = y_true.shape[0]

    if y_pred[0:N//2].max() <= y_pred[N//2:N].min(): # perfectly separable case
        return (y_pred[0:N//2].max() + y_pred[N//2:N].min()) / 2 

    best_acc = 0 
    best_thres = 0 
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp>=thres] = 1
        temp[temp<thres] = 0

        acc = (temp == y_true).sum() / N  
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc 
    
    return best_thres, best_acc

def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc

if __name__ == "__main__":
    train_dataset_name = "prithivMLmods/Context-Based-Chat-Summary-Plus"
    eval_dataset_name = "prithivMLmods/Context-Based-Chat-Summary-Base"
    padding_side = "right"

    train_dataset = load_dataset(train_dataset_name, split="train[0:500]", cache_dir="/scratch/vidit_a_mfs.iitr/hf_cache")
    eval_dataset = load_dataset(eval_dataset_name, split="train[0:500]", cache_dir="/scratch/vidit_a_mfs.iitr/hf_cache")
    # generated_train_dataset = train_dataset.map(generate_prompt, remove_columns=list(train_dataset.features))
    # generated_val_dataset = eval_dataset.map(generate_prompt, remove_columns=list(eval_dataset.features))

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", trust_remote_code=True, cache_dir="/scratch/vidit_a_mfs.iitr/hf_cache")
    tokenizer.padding_side = padding_side
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    # tokenizer.add_bos_token, tokenizer.add_eos_token

    train_ratios = []
    eval_ratios = []
    num_fail = 0
    for i, sample in enumerate(train_dataset):
        if i < 500:
            true_outs = tokenizer.encode(sample["headlines"], return_tensors="pt")
            finetune_probs = torch.load(f"/scratch/vidit_a_mfs.iitr/finetune-and-attack/probs/finetune/fullbits/train/{i}.pt")
            base_probs = torch.load(f"/scratch/vidit_a_mfs.iitr/finetune-and-attack/probs/base/fullbits/train/{i}.pt")
            if true_outs.shape[-1] < min(finetune_probs.shape[0], base_probs.shape[0]):
                finetune_probs = finetune_probs[:true_outs.shape[-1]]
                base_probs = base_probs[:true_outs.shape[-1]]
            if true_outs.shape[-1] > min(finetune_probs.shape[0], base_probs.shape[0]):
                true_outs = true_outs[:, :min(finetune_probs.shape[0], base_probs.shape[0])]
                finetune_probs = finetune_probs[:true_outs.shape[-1]]
                base_probs = base_probs[:true_outs.shape[-1]]
            pp1 = Perplexity()
            try:
                pp1.update(finetune_probs.unsqueeze(0), true_outs)
                finetune_pp = pp1.compute()
                pp2 = Perplexity()
                pp2.update(base_probs.unsqueeze(0), true_outs)
                base_pp = pp2.compute()
                train_ratios.append(base_pp / finetune_pp)
            except:
                num_fail += 1
                print(f"Failed for {i}")
                continue
    print(f"Failed for {num_fail} samples")
        
    torch.save(torch.Tensor(train_ratios), "/scratch/vidit_a_mfs.iitr/finetune-and-attack/ratios/fullbits_train.pt")

    for j, sample in enumerate(eval_dataset):
        true_outs = tokenizer.encode(sample["response"], return_tensors="pt")
        finetune_probs = torch.load(f"/scratch/vidit_a_mfs.iitr/finetune-and-attack/probs/finetune/fullbits/eval/{j}.pt")
        base_probs = torch.load(f"/scratch/vidit_a_mfs.iitr/finetune-and-attack/probs/base/fullbits/eval/{j}.pt")
        if len(base_probs.shape) == 1:
            base_probs = base_probs.unsqueeze(0)
        if true_outs.shape[-1] < min(finetune_probs.shape[0], base_probs.shape[0]):
            finetune_probs = finetune_probs[:true_outs.shape[-1]]
            base_probs = base_probs[:true_outs.shape[-1]]
        if true_outs.shape[-1] > min(finetune_probs.shape[0], base_probs.shape[0]):
            true_outs = true_outs[:, :min(finetune_probs.shape[0], base_probs.shape[0])]
            finetune_probs = finetune_probs[:true_outs.shape[-1]]
            base_probs = base_probs[:true_outs.shape[-1]]
        pp3 = Perplexity()
        try:
            pp3.update(finetune_probs.unsqueeze(0), true_outs)
            finetune_pp = pp3.compute()
            pp4 = Perplexity()
            pp4.update(base_probs.unsqueeze(0), true_outs)
            base_pp = pp4.compute()
            eval_ratios.append(base_pp / finetune_pp)
        except:
            num_fail += 1
            print(f"Failed for {j}")
            continue
    print(f"Failed for {num_fail} samples")
        
    torch.save(torch.Tensor(eval_ratios), "/scratch/vidit_a_mfs.iitr/finetune-and-attack/ratios/fullbits_eval.pt")
    
    pred_list_1 = []
    pred_list_1.extend(eval_ratios[:150])
    pred_list_1.extend(train_ratios[:150])
    true_list_1 = []
    true_list_1.extend([0] * 150)
    true_list_1.extend([1] * 150)
    best_thres, best_acc = find_best_threshold(np.array(true_list_1), np.array(pred_list_1))
    
    print(f"Best threshold: {best_thres}, best accuracy: {best_acc}")

    pred_list_2 = []
    pred_list_2.extend(eval_ratios[150:200])
    pred_list_2.extend(train_ratios[150:200])
    true_list_2 = []
    true_list_2.extend([0] * 50)
    true_list_2.extend([1] * 50)

    print(np.array(true_list_2).shape, np.array(pred_list_2).shape, best_thres)

    r_acc, f_acc, acc = calculate_acc(np.array(true_list_2), np.array(pred_list_2), best_thres)
    print(f"Real accuracy: {r_acc}, Fake accuracy: {f_acc}, Total accuracy: {acc}")

    ap = average_precision_score(np.array(true_list_2), np.array(pred_list_2))
    print(f"Average precision: {ap}")