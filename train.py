#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import datasets
from datasets import Dataset
from datasets import load_dataset
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, LongformerConfig
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
#import wandb
import os


# In[3]:


# define accuracy metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # argmax(pred.predictions, axis=1)
    #pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# In[4]:


train_data = torch.load('./data/train_data.pt')
validation_data = torch.load('./data/validation_data.pt')
loader = DataLoader(validation_data, batch_size=1, shuffle=True)
it = iter(loader)


# In[5]:


# define the training arguments
training_args = TrainingArguments(
    output_dir = './results',
    num_train_epochs = 5,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 8,    
    per_device_eval_batch_size= 2,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    disable_tqdm = False, 
    load_best_model_at_end=True,
    warmup_steps=200,
    weight_decay=0.01,
    logging_steps = 4,
    fp16 = False,
    logging_dir='./logs',
    dataloader_num_workers = 0,
    run_name = 'longformer-classification-no-tuning'
)


# In[6]:


# load model and tokenizer and define length of the text sequence
model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                           gradient_checkpointing=False,
                                                           attention_window = 512)


# In[7]:


# instantiate the trainer class and check for available devices
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=validation_data
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


# In[8]:


os.environ["WANDB_DISABLED"] = "true"


# In[ ]:


# train the model
trainer.train()


# In[ ]:


# save the best model
trainer.save_model('./results/polarity-no-finetuning')


# In[ ]:


trainer.evaluate()

