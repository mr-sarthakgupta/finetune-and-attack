import os, torch, wandb
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging

# wandb.login(key = '3258cca5528aa7bbd0faa372e4d4fdbe57ff959b')
run = wandb.init(
    project='mistral_dolly-1b', 
    job_type="training", 
    anonymous="allow"
)

base_model = "mistralai/Mistral-7B-v0.3"
dataset_name = "databricks/databricks-dolly-15k"
new_model = "mistral-7b-dolly-1-epoch"
padding_side = "right"

train_dataset = load_dataset(dataset_name, split="train[0:800]", cache_dir="/scratch/sarthak_g.iitr/hf_cache")
eval_dataset = load_dataset(dataset_name, split="train[800:1000]", cache_dir="/scratch/sarthak_g.iitr/hf_cache")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, cache_dir="/scratch/sarthak_g.iitr/hf_cache")
tokenizer.padding_side = padding_side
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

# Helper function to format the prompt
def generate_prompt(sample):
    full_prompt =f"""<s>[INST]{sample['instruction']}
{f"Here is some context: {sample['context']}" if len(sample["context"]) > 0 else None}
 [/INST] {sample['response']}
</s>"""
    return {"text": full_prompt}

generated_train_dataset = train_dataset.map(generate_prompt, remove_columns=list(train_dataset.features))
generated_val_dataset = eval_dataset.map(generate_prompt, remove_columns=list(train_dataset.features))

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# Load base model (Mistral 7B)
bnb_config = BitsAndBytesConfig(  
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)
model = AutoModelForCausalLM.from_pretrained(
        base_model,
        # load_in_4bit=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        cache_dir="/scratch/sarthak_g.iitr/hf_cache"
)
model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

#Adding the adapters in the layers
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1, # Coventional
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
)
model = get_peft_model(model, peft_config)
print_trainable_parameters(model)

# Hyperparamter
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=1.8e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="wandb",
    evaluation_strategy="steps", # Evaluate the model every logging step
    eval_steps=25,               # Evaluate and save checkpoints every x steps
    do_eval=True,                # Perform evaluation at the end of training
)

# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=generated_train_dataset,
    eval_dataset=generated_val_dataset,
    peft_config=peft_config,
    max_seq_length=None,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

# Begin training
trainer.train()
print('training complete, meow')

# Save the fine-tuned lora model
trainer.model.save_pretrained(new_model)
wandb.finish()
model.config.use_cache = True
model.eval()

# use base model directly from HF for production i.e. mistralai/Mistral-7B-v0.1
try:
    trainer.model.push_to_hub(new_model, use_temp_dir=False)
except:
    print("An exception occurred")

logging.set_verbosity(logging.CRITICAL)

prompt = """
What is a Plumbus? Here is some context: Plumbuses are made of organic tissue, fleebs, dinglebops, and grumbos.
"""
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, eos_token_id=model.config.eos_token_id, max_new_tokens=25)
result = pipe(f"<s>[INST] {prompt} [/INST]")
generated = result[0]['generated_text']
print(generated[generated.find('[/INST]')+8:])

# Empty VRAM
del model
del pipe
del trainer

# Reload model in FP16 and merge it with LoRA weights
basemodel = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir="/scratch/sarthak_g.iitr/hf_cache"
)
#model = PeftModel.from_pretrained(basemodel, new_model) if you pushed lora to HF
model = PeftModel.from_pretrained(basemodel, './results/checkpoint-200', cache_dir="/scratch/sarthak_g.iitr/hf_cache")
model = model.merge_and_unload() # Merge lora back to base model

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, cache_dir="/scratch/sarthak_g.iitr/hf_cache")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = padding_side

# use base model directly from HF for production i.e. mistralai/Mistral-7B-v0.1
try:
    model.push_to_hub(new_model + "-merged", max_shard_size='2GB')
    tokenizer.push_to_hub(new_model + "-merged")
except:
    print("An exception occurred")