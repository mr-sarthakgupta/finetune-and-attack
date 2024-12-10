## Finetune & Attack

This repository contains code for fine-tuning a mistral-7b model on the databricks-dolly-15k dataset and then attacking it for membership inference to test the privacy leakage in the model. We choose the attack described in the paper TRAINING DATA LEAKAGE ANALYSIS IN LANGUAGE MODELS([arxiv](https://arxiv.org/abs/2101.05405)). In particular, this attack is chosen because of it's relevance to the paradigm of LLM fine-tuning as often the pre-trained model is publically available while the data used for fine-tuning could be proprietary.

The models are available at:
- Mistral-7B-v0.1: mrsarthakgupta/mistral-v0.1-7b-dolly-3-epoch
- Mistral-7B-v0.3: mrsarthakgupta/mistral-v0.3-7b-dolly-1-epoch
