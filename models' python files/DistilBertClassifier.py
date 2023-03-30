# ===========================================
# ||                                       ||
# ||Section 1: Importing modules           ||
# ||                                       ||
# ===========================================

import transformers
from datasets import load_dataset, load_metric,  Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os
import nltk
import torch
import evaluate
import sys
import pandas as pd
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
import os
import torch.cuda as cuda
import gc
import optuna
from utils4gpu import *

# ===========================================
# ||                                       ||
# ||Section 2: Utlis 4 DistilBert          ||
# ||                                       ||
# ===========================================

# getting the max length of the tokenized tweet
def getting_max_length(tokenizer, items):

  # initialize a list for lengths
  max_len = 0

  # iterate over the list
  for item in items:

    # record lenght of each item
    lenght_of_item = len(tokenizer.encode(item, add_special_tokens = True))

    if lenght_of_item > max_len:

      max_len = lenght_of_item

  return max_len

# ===========================================
# ||                                       ||
# ||Section 3: checking gpu, choosing      ||
# ||             device, and model         ||
# ||                                       ||
# ===========================================

# CHECK IF GPU IS UP
check_gpu_availability()

# SAVE THE DEVICE WE ARE WORKING WITH
device = getting_device(gpu_prefence=True)

# SHOULD BE FEW MB
print_gpu_utilization()

# SETTING HF CHECKPOINT/MODEL
model_nm = "distilbert-base-uncased"

# ===========================================
# ||                                       ||
# ||Section 4: Importing doc and split     ||
# ||                                       ||
# ===========================================

# Read csv files to create pandas dataframes
path2test = '/content/drive/MyDrive/ML_proj/zaazazza/Copia de test_df.csv'
test_df = pd.read_csv(path2test)

path2val = '/content/drive/MyDrive/ML_proj/zaazazza/Copia de validation_df.csv'
validation_df = pd.read_csv(path2val)

path2train = '/content/drive/MyDrive/ML_proj/zaazazza/Copia de train_df.csv'
train_df = pd.read_csv(path2train)

# Renaming columns
train_df.rename(columns = {"target":"labels"}, inplace = True)
validation_df.rename(columns = {"target":"labels"}, inplace = True)
test_df.rename(columns = {"target":"labels"}, inplace = True)

# pandas2dataset
ds_train = Dataset.from_pandas(train_df)
ds_validation = Dataset.from_pandas(validation_df)
ds_test = Dataset.from_pandas(test_df)

# ===========================================
# ||                                       ||
# ||Section 5: tokenization, tensorization ||
# ||              and collider             ||
# ||                                       ||
# ===========================================

# IMPORTING OUR TOKENIZER
tokz = AutoTokenizer.from_pretrained(model_nm)

# GETTING THE LENGHT MAX
max_len = getting_max_length(tokz,ds_train["text"])

# DEFINING A TOKENIZE FUNCTION TO TOKENIZE BOTH THE TWO DATASETS
def tok_func(x): return tokz(x["text"], truncation=True, padding = "max_length", max_length=max_len)

# TOKENIZING THE DS
tok_ds_train = ds_train.map(tok_func, batched=True, remove_columns=['text','id', 'keyword', 'location'])
tok_ds_validation = ds_validation.map(tok_func, batched=True, remove_columns=['text','id', 'keyword', 'location'])
tok_ds_test = ds_test.map(tok_func, batched=True, remove_columns=['text','id', 'keyword', 'location'])

# CREATE A DATASET TO FEED THE MODEL
ds = DatasetDict({"train":tok_ds_train,
                  "validation":tok_ds_validation,
             "test": tok_ds_test})

# GETTING THE COLLATOR
data_collator = DataCollatorWithPadding(tokenizer=tokz)

# ===========================================
# ||                                       ||
# ||Section 6: metrics
# ||                                       ||
# ===========================================

# 1) F1 and ACCURACY

# now that we have our callable object, we define a function that the trainer can use to compute its metric => we cannot call directly metrics.compute because the output
# of the model is not a prediction but a logist
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}

# ===========================================
# ||                                       ||
# ||Section 7: hyperparam search           ||
# ||                                       ||
# ===========================================


num_r = 1

# Define the search space for hyperparameters using Optuna's distributions.
def objective(trial):
    global num_r

    # Rename folder containing the old model
    if os.path.exists("/content/distilbert-base-uncased"):
      os.rename("/content/distilbert-base-uncased", os.path.join(os.path.dirname("/content/distilbert-base-uncased"), str(num_r)))
    num_r += 1

    # IMPORTING THE MODEL
    model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels = 2).to(device)
    # checking if the model is on the gpu
    print_gpu_utilization()

    # Hyperparameters to optimize
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-4)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    num_train_epochs = trial.suggest_int('num_train_epochs', 1, 5)

    # setting the hyperparameter for the trainer
    training_args = TrainingArguments(
        model_nm,
        evaluation_strategy = "epoch",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps = 50, # FROM BELOW MEMORY TRICKS
        gradient_accumulation_steps=16, # adding them to offset small batch size due to memory problem => so 2*8 => 16 batch-size traning
        fp16 = True
    )

    # passing in the hyperparameter for the trainer
    trainer = Trainer(
        model = model, # our model
        args = training_args, # hyperparameter defined before
        train_dataset = ds["train"],
        eval_dataset = ds["validation"],
        compute_metrics = compute_metrics, # evaluation function defined before
        data_collator = data_collator,
    )

    # TRAINING LOOP
    print(" ")
    print("START TRAINING ")
    print(" ")
    trainer.train()
    print("DONE TRAINING")

    # TESTING
    print(" ")
    print("START VALIDATION")
    print(" ")
    predictions = trainer.predict(ds["validation"])
    eval_result = compute_metrics(predictions)
    print(eval_result)
    print("DONE VALIDATION")

    # Return the evaluation metric to be optimized by Optuna.
    return 1 - eval_result['accuracy']

# Define the Optuna study and run the hyperparameter search.
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=2)
print(study.best_params)
best_trial = study.best_trial

# Train the model with the best hyperparameters found by optuna and evaluate it on the test data.
best_num_train_epochs = best_trial.params['num_train_epochs']
best_weight_decay = best_trial.params['weight_decay']
best_learning_rate = best_trial.params['learning_rate']


# ===========================================
# ||                                       ||
# ||Section 8: the model and hyperparam    ||
# ||                                       ||
# ===========================================

# Rename folder containing the old model
if os.path.exists("/content/distilbert-base-uncased"):
    os.rename("/content/distilbert-base-uncased", os.path.join(os.path.dirname("/content/distilbert-base-uncased"), '1'))

# IMPORTING THE MODEL
model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels = 2).to(device)
# checking if the model is on the gpu
print_gpu_utilization()

# setting the hyperparameter for the trainer
training_args = TrainingArguments(
    model_nm,
    evaluation_strategy = "epoch",
    learning_rate=best_learning_rate,
    weight_decay=best_weight_decay,
    num_train_epochs=best_num_train_epochs,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps = 50, # FROM BELOW MEMORY TRICKS
    gradient_accumulation_steps=16, # adding them to offset small batch size due to memory problem => so 2*8 => 16 batch-size traning
    fp16 = True
    )

# passing in the hyperparameter for the trainer
trainer = Trainer(
    model = model, # our model
    args = training_args, # hyperparameter defined before
    train_dataset = ds["train"],
    eval_dataset = ds["validation"],
    compute_metrics = compute_metrics, # evaluation function defined before
    data_collator = data_collator,
)

# ===========================================
# ||                                       ||
# ||Section 9: training and testing        ||
# ||                                       ||
# ===========================================

# TRAINING LOOP
print(" ")
print("START TRAINING ")
print(" ")
trainer.train()
print("DONE TRAINING")

# TESTING
print(" ")
print("\033[32mSTARTING FINAL TESTING OF THE MODEL\033[0m")
print(" ")
predictions = trainer.predict(ds["test"])
eval_result = compute_metrics(predictions)
print(eval_result)
print("DONE TESTING")

# ===========================================
# ||                                       ||
# ||Section 10: validation and bias        ||
# ||                                       ||
# ===========================================

from datasets import concatenate_datasets
from sklearn.model_selection import KFold

ds = concatenate_datasets([tok_ds_train, tok_ds_validation, tok_ds_test])

n=5
kf = KFold(n_splits=n, random_state=42, shuffle=True)

accuracy = []
f1 = []
set1 = train_df
set1.rename(columns = {"target":"labels"}, inplace = True)
i = 0
for train_index, val_index in kf.split(set1):
  i+=231
  if os.path.exists("/content/distilbert-base-uncased"):
     os.rename("/content/distilbert-base-uncased", os.path.join(os.path.dirname("/content/distilbert-base-uncased"), str(i)))
  # splitting Dataframe (dataset not included)
  train_df = set1.iloc[train_index]
  val_df = set1.iloc[val_index]
  ds_train = Dataset.from_pandas(train_df)
  ds_validation = Dataset.from_pandas(val_df)
  tok_ds_train = ds_train.map(tok_func, batched=True, remove_columns=['text','id', 'keyword', 'location'])
  tok_ds_validation = ds_validation.map(tok_func, batched=True, remove_columns=['text','id', 'keyword', 'location'])
  ds = DatasetDict({"train":tok_ds_train, "validation":tok_ds_validation})

  # cleaning gpu and loading the model
  clean_gpu()
  model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels = 2).to(device)
  # setting up the trainer
  trainer = Trainer(model = model, args = training_args, train_dataset = ds["train"], eval_dataset = ds["validation"], compute_metrics = compute_metrics, data_collator = data_collator)
  # train the model
  trainer.train()
  # access the performance
  eval_accuracy = trainer.evaluate(ds["validation"])['eval_accuracy']
  eval_f1 = trainer.evaluate(ds["validation"])['eval_f1']
  # append model score
  f1.append(eval_f1)
  accuracy.append(eval_accuracy)


# ===========================================
# ||                                       ||
# ||Section 11: saving the model           ||
# ||                                       ||
# ===========================================

import os
# Set the output directory
output_dir = '/content/output/DISTILBERT'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the model and tokenizer to the output directory
trainer.save_model(output_dir)
tokz.save_pretrained(output_dir)