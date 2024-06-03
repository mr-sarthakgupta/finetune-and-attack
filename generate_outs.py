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

    train_dataset = load_dataset(dataset_name, split="train[0:800]", cache_dir="/scratch/sarthak_g.iitr/hf_cache")
    eval_dataset = load_dataset(dataset_name, split="train[800:1000]", cache_dir="/scratch/sarthak_g.iitr/hf_cache")
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3", trust_remote_code=True, cache_dir="/scratch/sarthak_g.iitr/hf_cache")
    tokenizer.padding_side = padding_side
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    tokenizer.add_bos_token, tokenizer.add_eos_token

    # Load base model (Mistral 7B)
    bnb_config = BitsAndBytesConfig(  
        load_in_4bit= True,
        bnb_4bit_quant_type= "nf4",
        bnb_4bit_compute_dtype= torch.bfloat16,
        bnb_4bit_use_double_quant= False,
    )

    mistral_dolly = AutoModelForCausalLM.from_pretrained(
            "mrsarthakgupta/mistral-v0.3-7b-dolly-1-epoch",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
            cache_dir="/scratch/sarthak_g.iitr/hf_cache"
    )
    
    
    mistral_dolly.config.use_cache = True
    mistral_dolly.eval()
    
    for i, datum in enumerate(train_dataset):
        if i < 200:
            print(f"finetune/train/{i}")
            if len(datum['context']) > 0:
                input_tokens = tokenizer.encode(f"<s>[INST] {datum['instruction']}" + f"Here is some context: {datum['context']} [/INST]", return_tensors="pt")
                output_dict = mistral_dolly.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=50, return_dict_in_generate=True)
                probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
            else:
                input_tokens = tokenizer.encode(f"<s>[INST] {datum['instruction']}[/INST]", return_tensors="pt")
                output_dict = mistral_dolly.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=50, return_dict_in_generate=True)
                probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
            torch.save(probs.cpu(), f"/scratch/sarthak_g.iitr/dynamo/probs/finetune/train/{i}.pt")

    for j, datum in enumerate(eval_dataset):
        print(f"finetune/eval/{j}")
        if len(datum['context']) > 0:
            input_tokens = tokenizer.encode(f"<s>[INST] {datum['instruction']}" + f"Here is some context: {datum['context']} [/INST]", return_tensors="pt")
            output_dict = mistral_dolly.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=50, return_dict_in_generate=True)
            probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
        else:
            input_tokens = tokenizer.encode(f"<s>[INST] {datum['instruction']}[/INST]", return_tensors="pt")
            output_dict = mistral_dolly.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=50, return_dict_in_generate=True)
            probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
        torch.save(probs.cpu(), f"/scratch/sarthak_g.iitr/dynamo/probs/finetune/eval/{j}.pt")

    del mistral_dolly # remove fine-tuned model to make space in vRAM

    # base_mistral = AutoModelForCausalLM.from_pretrained(
    #     "mistralai/Mistral-7B-v0.3",
    #     # load_in_4bit=True,
    #     quantization_config=bnb_config,
    #     torch_dtype=torch.bfloat16,
    #     device_map="cuda:0",
    #     trust_remote_code=True,
    #     cache_dir="/scratch/sarthak_g.iitr/hf_cache"
    # )
    
    # base_mistral.config.use_cache = True
    # base_mistral.eval()
    
    # for k, datum in enumerate(train_dataset):
    #     if k < 200:
    #         print(f"base/train/{k}")
    #         if len(datum['context']) > 0:
    #             input_tokens = tokenizer.encode(f"<s>[INST] {datum['instruction']}" + f"Here is some context: {datum['context']} [/INST]", return_tensors="pt")
    #             output_dict = base_mistral.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=50, return_dict_in_generate=True)
    #             probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
    #         else:
    #             input_tokens = tokenizer.encode(f"<s>[INST] {datum['instruction']}[/INST]", return_tensors="pt")
    #             output_dict = base_mistral.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=50, return_dict_in_generate=True)
    #             probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
    #         torch.save(probs.cpu(), f"/scratch/sarthak_g.iitr/dynamo/probs/base/train/{k}.pt")

    # for l, datum in enumerate(eval_dataset):
    #     print(f"finetune/eval/{l}")
    #     if len(datum['context']) > 0:
    #         input_tokens = tokenizer.encode(f"<s>[INST] {datum['instruction']}" + f"Here is some context: {datum['context']} [/INST]", return_tensors="pt")
    #         output_dict = base_mistral.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=50, return_dict_in_generate=True)
    #         probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
    #     else:
    #         input_tokens = tokenizer.encode(f"<s>[INST] {datum['instruction']}[/INST]", return_tensors="pt")
    #         output_dict = base_mistral.generate(input_tokens.to('cuda'), output_scores = True, max_new_tokens=50, return_dict_in_generate=True)
    #         probs = torch.stack(output_dict['scores'], dim = 0).squeeze()
    #     torch.save(probs.cpu(), f"/scratch/sarthak_g.iitr/dynamo/probs/base/eval/{l}.pt")