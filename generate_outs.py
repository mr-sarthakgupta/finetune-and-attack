import torch
from torcheval.metrics import Perplexity
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def generate_prompt(sample):
    messages = f"you are a chatbot for summarizing text and form headlines, the user will provide you with a text and you have to summarize it and present a headline only and nothing else, don't add any other tokens, just the headline. Here is the text to summarize:  {sample['text']}"
    return messages

if __name__ == "__main__":
    dataset_name = "prithivMLmods/Context-Based-Chat-Summary-Plus"
    padding_side = "right"

    train_dataset = load_dataset(dataset_name, split="train[0:1000]", cache_dir="/scratch/vidit_a_mfs.iitr/hf_cache")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", cache_dir="/scratch/vidit_a_mfs.iitr/hf_cache")
    tokenizer.padding_side = padding_side
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    # tokenizer.add_bos_token, tokenizer.add_eos_token

    mistral_dolly = AutoModelForCausalLM.from_pretrained(
            "prithivMLmods/Llama-Chat-Summary-3.2-3B",
            # load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            # trust_remote_code=True,
            cache_dir="/scratch/vidit_a_mfs.iitr/hf_cache"
    )

    mistral_dolly.config.use_cache = True
    mistral_dolly.eval()
    
    for i, datum in enumerate(train_dataset):
        if i < 500:
            print(f"train {i}")
            input_tokens = tokenizer(generate_prompt(datum), return_tensors="pt")['input_ids']
            output_dict = mistral_dolly.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=35, return_dict_in_generate=True)
            probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
            torch.save(probs.cpu(), f"/scratch/vidit_a_mfs.iitr/finetune-and-attack/probs/finetune/quantized/train/{i}.pt")
            del probs, output_dict, input_tokens
            torch.cuda.empty_cache()
    
    torch.cuda.empty_cache()


    del mistral_dolly # remove fine-tuned model to make space in vRAM
    torch.cuda.empty_cache()

    mistral_dolly_unquantized = AutoModelForCausalLM.from_pretrained(
            "prithivMLmods/Llama-Chat-Summary-3.2-3B",
            torch_dtype=torch.float32,
            device_map="cuda:0",
            cache_dir="/scratch/vidit_a_mfs.iitr/hf_cache"
    )

    mistral_dolly_unquantized.config.use_cache = True
    mistral_dolly_unquantized.eval()
    
    for i, datum in enumerate(train_dataset):
        if i < 500:
            print(f"train {i}")
            input_tokens = tokenizer(generate_prompt(datum), return_tensors="pt")['input_ids']
            output_dict = mistral_dolly_unquantized.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=35, return_dict_in_generate=True)
            probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
            torch.save(probs.cpu(), f"/scratch/vidit_a_mfs.iitr/finetune-and-attack/probs/finetune/fullbits/train/{i}.pt")
            del probs, output_dict, input_tokens
            torch.cuda.empty_cache()
    

    del mistral_dolly_unquantized # remove fine-tuned model to make space in vRAM
    torch.cuda.empty_cache()

    base_mistral = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        torch_dtype=torch.float16,
        device_map="cuda:0",
        # trust_remote_code=True,
        cache_dir="/scratch/vidit_a_mfs.iitr/hf_cache"
    )
    
    base_mistral.config.use_cache = True
    base_mistral.eval()
    
    for k, datum in enumerate(train_dataset):
        if k < 500:
            print(f"base/train/{k}")
            input_tokens = tokenizer(generate_prompt(datum), return_tensors="pt")['input_ids']
            output_dict = base_mistral.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=35, return_dict_in_generate=True)
            probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
            torch.save(probs.cpu(), f"/scratch/vidit_a_mfs.iitr/finetune-and-attack/probs/base/quantized/train/{k}.pt")
            del probs, output_dict, input_tokens
            torch.cuda.empty_cache()
     
    del base_mistral 
    torch.cuda.empty_cache()

    base_mistral_unquantized = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        torch_dtype=torch.float32,
        device_map="cuda:0",
        trust_remote_code=True,
        cache_dir="/scratch/vidit_a_mfs.iitr/hf_cache"
    )
    
    base_mistral_unquantized.config.use_cache = True
    base_mistral_unquantized.eval()
    
    for k, datum in enumerate(train_dataset):
        if k < 500:
            print(f"base/train/{k}")
            input_tokens = tokenizer(generate_prompt(datum), return_tensors="pt")
            output_dict = base_mistral_unquantized.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=50, return_dict_in_generate=True)
            probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
            torch.save(probs.cpu(), f"/scratch/vidit_a_mfs.iitr/finetune-and-attack/probs/base/fullbits/train/{k}.pt")
            del probs, output_dict, input_tokens
            torch.cuda.empty_cache()
