import torch
from torcheval.metrics import Perplexity
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def generate_prompt(sample):
    full_prompt =f"""<s>[INST]{sample['instruction']}
    {f"Here is some context: {sample['context']}" if len(sample["context"]) > 0 else None}
    [/INST] {sample['response']}
    </s>"""
    return {"text": full_prompt}

if __name__ == "__main__":
    dataset_name = "databricks/databricks-dolly-15k"
    padding_side = "right"

    train_dataset = load_dataset(dataset_name, split="train[0:800]", cache_dir="/scratch/a_singh4ee.iitr/hf_cache")
    eval_dataset = load_dataset(dataset_name, split="train[800:1000]", cache_dir="/scratch/a_singh4ee.iitr/hf_cache")
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3", cache_dir="/scratch/a_singh4ee.iitr/hf_cache")
    tokenizer.padding_side = padding_side
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    tokenizer.add_bos_token, tokenizer.add_eos_token


    mistral_dolly = AutoModelForCausalLM.from_pretrained(
            "mrsarthakgupta/mistral-v0.3-7b-dolly-1-epoch",
            # load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            # trust_remote_code=True,
            cache_dir="/scratch/a_singh4ee.iitr/hf_cache"
    )

    mistral_dolly.config.use_cache = True
    mistral_dolly.eval()
    
    for i, datum in enumerate(train_dataset):
        if i < 200:
            print(f"train {i}")
            if len(datum['context']) > 0:
                input_tokens = tokenizer.encode(f"<s>[INST] {datum['instruction']}" + f"Here is some context: {datum['context']} [/INST]", return_tensors="pt")
                output_dict = mistral_dolly.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=50, return_dict_in_generate=True)
                probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
            else:
                input_tokens = tokenizer.encode(f"<s>[INST] {datum['instruction']}[/INST]", return_tensors="pt")
                output_dict = mistral_dolly.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=50, return_dict_in_generate=True)
                probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
            torch.save(probs.cpu(), f"/scratch/a_singh4ee.iitr/finetune-and-attack/probs/finetune/quantized/train/{i}.pt")
            del probs, output_dict, input_tokens
            torch.cuda.empty_cache()
    
    torch.cuda.empty_cache()

    for j, datum in enumerate(eval_dataset):
        print(f"eval {j}")
        if len(datum['context']) > 0:
            input_tokens = tokenizer.encode(f"<s>[INST] {datum['instruction']}" + f"Here is some context: {datum['context']} [/INST]", return_tensors="pt")
            output_dict = mistral_dolly.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=50, return_dict_in_generate=True)
            probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
        else:
            input_tokens = tokenizer.encode(f"<s>[INST] {datum['instruction']}[/INST]", return_tensors="pt")
            output_dict = mistral_dolly.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=50, return_dict_in_generate=True)
            probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
        torch.save(probs.cpu(), f"/scratch/a_singh4ee.iitr/finetune-and-attack/probs/finetune/quantized/eval/{j}.pt")
        del probs, output_dict, input_tokens
        torch.cuda.empty_cache()
    

    exit()

    del mistral_dolly # remove fine-tuned model to make space in vRAM
    torch.cuda.empty_cache()

    mistral_dolly_unquantized = AutoModelForCausalLM.from_pretrained(
            "mrsarthakgupta/mistral-v0.3-7b-dolly-1-epoch",
            torch_dtype=torch.bfloat32,
            device_map="cuda:0",
            cache_dir="/scratch/a_singh4ee.iitr/hf_cache"
    )

    mistral_dolly_unquantized.config.use_cache = True
    mistral_dolly_unquantized.eval()
    
    for i, datum in enumerate(train_dataset):
        if i < 200:
            print(f"train {i}")
            if len(datum['context']) > 0:
                input_tokens = tokenizer.encode(f"<s>[INST] {datum['instruction']}" + f"Here is some context: {datum['context']} [/INST]", return_tensors="pt")
                output_dict = mistral_dolly_unquantized.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=50, return_dict_in_generate=True)
                probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
            else:
                input_tokens = tokenizer.encode(f"<s>[INST] {datum['instruction']}[/INST]", return_tensors="pt")
                output_dict = mistral_dolly_unquantized.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=50, return_dict_in_generate=True)
                probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
            torch.save(probs.cpu(), f"/scratch/a_singh4ee.iitr/finetune-and-attack/probs/finetune/fullbits/train/{i}.pt")
            del probs, output_dict, input_tokens
            torch.cuda.empty_cache()
    

    for j, datum in enumerate(eval_dataset):
        print(f"eval {j}")
        if len(datum['context']) > 0:
            input_tokens = tokenizer.encode(f"<s>[INST] {datum['instruction']}" + f"Here is some context: {datum['context']} [/INST]", return_tensors="pt")
            output_dict = mistral_dolly_unquantized.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=50, return_dict_in_generate=True)
            probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
        else:
            input_tokens = tokenizer.encode(f"<s>[INST] {datum['instruction']}[/INST]", return_tensors="pt")
            output_dict = mistral_dolly_unquantized.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=50, return_dict_in_generate=True)
            probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
        torch.save(probs.cpu(), f"/scratch/a_singh4ee.iitr/finetune-and-attack/probs/finetune/fullbits/eval/{j}.pt")
        del probs, output_dict, input_tokens
        torch.cuda.empty_cache()
    

    del mistral_dolly_unquantized # remove fine-tuned model to make space in vRAM

    base_mistral = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.3",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        # trust_remote_code=True,
        cache_dir="/scratch/a_singh4ee.iitr/hf_cache"
    )
    
    base_mistral.config.use_cache = True
    base_mistral.eval()
    
    for k, datum in enumerate(train_dataset):
        if k < 200:
            print(f"base/train/{k}")
            if len(datum['context']) > 0:
                input_tokens = tokenizer.encode(f"<s>[INST] {datum['instruction']}" + f"Here is some context: {datum['context']} [/INST]", return_tensors="pt")
                output_dict = base_mistral.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=50, return_dict_in_generate=True)
                probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
            else:
                input_tokens = tokenizer.encode(f"<s>[INST] {datum['instruction']}[/INST]", return_tensors="pt")
                output_dict = base_mistral.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=50, return_dict_in_generate=True)
                probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
            torch.save(probs.cpu(), f"/scratch/a_singh4ee.iitr/finetune-and-attack/probs/base/quantized/train/{k}.pt")
            del probs, output_dict, input_tokens
            torch.cuda.empty_cache()
    

    for l, datum in enumerate(eval_dataset):
        print(f"finetune/eval/{l}")
        if len(datum['context']) > 0:
            input_tokens = tokenizer.encode(f"<s>[INST] {datum['instruction']}" + f"Here is some context: {datum['context']} [/INST]", return_tensors="pt")
            output_dict = base_mistral.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=50, return_dict_in_generate=True)
            probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
        else:
            input_tokens = tokenizer.encode(f"<s>[INST] {datum['instruction']}[/INST]", return_tensors="pt")
            output_dict = base_mistral.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=50, return_dict_in_generate=True)
            probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
        torch.save(probs.cpu(), f"/scratch/a_singh4ee.iitr/finetune-and-attack/probs/base/quantized/eval/{l}.pt")
        del probs, output_dict, input_tokens
        torch.cuda.empty_cache()
    



    base_mistral_unquantized = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.3",
        torch_dtype=torch.bfloat32,
        device_map="cuda:0",
        trust_remote_code=True,
        cache_dir="/scratch/a_singh4ee.iitr/hf_cache"
    )
    
    base_mistral_unquantized.config.use_cache = True
    base_mistral_unquantized.eval()
    
    for k, datum in enumerate(train_dataset):
        if k < 200:
            print(f"base/train/{k}")
            if len(datum['context']) > 0:
                input_tokens = tokenizer.encode(f"<s>[INST] {datum['instruction']}" + f"Here is some context: {datum['context']} [/INST]", return_tensors="pt")
                output_dict = base_mistral_unquantized.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=50, return_dict_in_generate=True)
                probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
            else:
                input_tokens = tokenizer.encode(f"<s>[INST] {datum['instruction']}[/INST]", return_tensors="pt")
                output_dict = base_mistral_unquantized.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=50, return_dict_in_generate=True)
                probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
            torch.save(probs.cpu(), f"/scratch/a_singh4ee.iitr/finetune-and-attack/probs/base/fullbits/train/{k}.pt")
            del probs, output_dict, input_tokens
            torch.cuda.empty_cache()
    

    for l, datum in enumerate(eval_dataset):
        print(f"finetune/eval/{l}")
        if len(datum['context']) > 0:
            input_tokens = tokenizer.encode(f"<s>[INST] {datum['instruction']}" + f"Here is some context: {datum['context']} [/INST]", return_tensors="pt")
            output_dict = base_mistral_unquantized.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=50, return_dict_in_generate=True)
            probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
        else:
            input_tokens = tokenizer.encode(f"<s>[INST] {datum['instruction']}[/INST]", return_tensors="pt")
            output_dict = base_mistral_unquantized.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=50, return_dict_in_generate=True)
            probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
        torch.save(probs.cpu(), f"/scratch/a_singh4ee.iitr/finetune-and-attack/probs/base/fullbits/eval/{l}.pt")
        del probs, output_dict, input_tokens
        torch.cuda.empty_cache()