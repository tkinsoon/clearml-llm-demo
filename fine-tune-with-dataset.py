#!/usr/bin/env python
# coding: utf-8

# # 2.1. Fine Tune with HF Dataset

# <img src="imgs/finetune_pipe.png" width="800" />

# In[1]:


#%env CLEARML_WEB_HOST=https://app.clearml.cnasg.dellcsc.com
#%env CLEARML_API_HOST=https://api.clearml.cnasg.dellcsc.com
#%env CLEARML_FILES_HOST=http://files.clearml.cnasg.dellcsc.com
# llm-workshop
#%env CLEARML_API_ACCESS_KEY=N3PGHCVPJ05SG9CXLO7X
#%env CLEARML_API_SECRET_KEY=z3EPKoHoY34q7GV9pmGIbF50WPdGIuXt0RNMQkFI6Fri1BRlwt
#%env CLEARML_LOG_MODEL=True
#%env CLEARML_VERIFY_CERTIFICATE=False
#%env CLEARML_CONFIG_FILE="/home/cnasg/codes/yx/clearml.conf"


# In[2]:


#!pip install -r requirements.txt
#!pip install clearml
get_ipython().system('pip install nbconvert')
from clearml import Task
task = Task.init(project_name='stk', task_name='fine-tune-llm')


# ### Installing and Importing Libraries

# In[3]:


import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer


# ## Load Pre-trained Model
# In order to fine-tune a model, we have to start off with the pre-trained model itself, and apply some configurations to prepare it for fine-tuning

# In[4]:


# Base model from huggingface
base_model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

# Fine-tuned model
new_model = "./results/tinyllama-medical"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map="auto"
)
model.config.use_cache = False
model.config.pretraining_tp = 1


# ## Load Tokenizer

# *In the context of natural language processing and machine learning, **pad_token_id** typically refers to the identifier or index assigned to a special token representing padding in a sequence. When working with sequences of varying lengths, it's common to pad shorter sequences with a special token to make them uniform in length.*
# 
# Eg:
# ```
# data = [ "I like cat", "Do you like cat too?"]
# tokenizer(data, padding=True, truncation=True, return_tensors='pt')
# ```
# Output:
# ```
# 'input_ids': tensor([[101,146,1176,5855,102,0,0,0],[101,2091,1128,1176,5855,1315,136,102]])
# 'token_type_ids': tensor([[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]])
# 'attention_mask': tensor([[1,1,1,1,1,0,0,0],[1,1,1,1,1,1,1,1]])
# ```

# In[5]:


# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# ## Load Dataset from 🤗HuggingFace
# 
# Hugging Face Datasets is designed to simplify the process of working with and accessing various datasets commonly used in NLP tasks. The library offers a collection of datasets for tasks such as text classification, named entity recognition, question answering, and more. [link](https://huggingface.co/datasets)

# In[6]:


# New instruction dataset
data = load_dataset("BI55/MedText", split='train')


# In[7]:


def prompt_formatter(dataset):
    dataset['Prompt'] = ' '.join(dataset['Prompt'].split())
    dataset['Completion'] = ' '.join(dataset['Completion'].split())

    full_prompt = f"<s>[INST] {dataset['Prompt']} [/INST] {dataset['Completion']} </s>"
    result = {}
    result['text'] = full_prompt
    return result


# In[8]:


formatted_dataset = data.map(prompt_formatter, remove_columns=['Prompt', 'Completion'])
print(formatted_dataset)
print(formatted_dataset[50]['text'])


# ## Prepare Model for Q-Lora INT4 Fine Tuning
# 
# <img src="imgs/qlora.png" width="800"/>
# 
# [Reference](https://www.linkedin.com/pulse/trends-llms-qlora-efficient-finetuning-quantized-vijay/?trk=article-ssr-frontend-pulse_more-articles_related-content-card)
# 
# Summary:
# 1. 4-bit quantization of the full pretrained language model to compress weights and reduce memory requirements using a novel NormalFloat encoding optimized for the distribution of neural network weights.
# 2. Low-rank adapters added densely throughout the layers of the 4-bit quantized base model. The adapters use full 16-bit precision and are finetuned while the base model remains fixed.
# 3. Double quantization further reduces memory by quantizing the first-stage quantization constants themselves from 32-bit to 8-bit with no accuracy loss.
# 4. Paged optimizers leverage unified memory to gracefully handle gradient checkpointing memory spikes and finetune models larger than the physical GPU memory through automatic paging.
# 5. Mixed precision combines 4-bit weights with 16-bit adapter parameters and activations, maximizing memory savings while retaining accuracy.

# ### LoRA Configuration
# 
# LoraConfig allows you to control how LoRA is applied to the base model through the following parameters:
# 
# ***target_modules** - The modules (for example, attention blocks) to apply the LoRA update matrices* [Reference](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms) - By default, all linear modules are targted.
# 

# In[9]:


model


# In[10]:


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

# Load LoRA configuration
peft_args = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(model, peft_args)
print_trainable_parameters(peft_model)


# In[11]:


# Set training parameters
training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="adamw_torch",
    save_steps=100,
    logging_steps=50,
    learning_rate=2e-4,
    weight_decay=0.001,
    max_grad_norm=0.3,
    max_steps=300,
    warmup_ratio=0.03,
    group_by_length=True,
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    peft_config=peft_args,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)


# In[ ]:


task.execute_remotely(queue_name="qw1")
# Train model
trainer.train()


# In[ ]:


new_model_adapter = "./results/tinyllama-medical-adapter"

# Save trained model
trainer.model.save_pretrained(new_model_adapter)
trainer.tokenizer.save_pretrained(new_model_adapter)


# ## Merging LoRa adapter weights into the base pre-trained model
# Now that we have successfully fine-tuned the model using LoRa, the next step is to merge the adapter weights into the original pre-trained model.
# We will be going through the steps in the next notebook. Before that, please restart or kill this current runtime to release the used memory from the GPU.

# In[15]:


task.close()


# In[ ]:




