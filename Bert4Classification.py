# ===========================================
# ||                                       ||
# ||Section 1: Importing modules           ||
# ||                                       ||
# ===========================================

import transformers
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os
import nltk
import torch
import evaluate
import sys
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
import torchvision.models as models
from utils4gpu import *


# ===========================================
# ||                                       ||
# ||Section 2: Utlis 4 Bert                ||
# ||                                       ||
# ===========================================

# FUNCTION 4 MAX LENGTH TO SET TOKENIZER MAX LENGTH

def get_list_of_lengths(text_column, tokenizer) -> int:
    token_lens = []

    for text in text_column:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens => split in symbolic/textual tokens and map them to integer ids
        tokens = tokenizer.encode(text, add_special_tokens=True)

        # checking the len of tokenized sentence
        token_lens.append(len(tokens))

    return token_lens


def get_max_lenghts(list_len) -> int:
    # PART 1 MAX

    # Convert the list to a PyTorch tensor
    tensor_data = torch.tensor(list_len)

    # getting the argmax index
    argmax_index = tensor_data.argmax().item()

    # getting the argmax

    argmax = list_len[argmax_index]
    print(f"The longest input sequence has value: {argmax}")

    # PART 2 HISTOGRAM

    # importing the library for the visualization
    import seaborn as sns

    # now we want to plot the histogram of the list of integers
    sns.histplot(list_len, bins=10)

    return argmax


# FUNCTION 4 GETTING THE DATASET IN  A FORM 4 PYTORCH

def convert_to_torch(item):
    input_ids = item['input_ids']
    token_type_ids = item['token_type_ids']
    attention_mask = item['attention_mask']
    label = item['labels']
    return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask, 'label': label}

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
model_nm = "bert-large-uncased"

# ===========================================
# ||                                       ||
# ||Section 4: Importing doc and split     ||
# ||                                       ||
# ===========================================

# Read in train and test CSV files using Pandas
path2train = '/content/drive/MyDrive/LT_SHARED_FOLDER/train.csv'
df = pd.read_csv(path2train)
# split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, validation_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Change the name of target values from "target" to "labels" to conform to BERT's standard
train_df.rename(columns = {"target":"labels"}, inplace = True)
validation_df.rename(columns = {"target":"labels"}, inplace = True)
test_df.rename(columns = {"target":"labels"}, inplace = True)

# pandas2dataset
ds_train = Dataset.from_pandas(train_df[["text","labels"]])
ds_validation = Dataset.from_pandas(validation_df[["text","labels"]])
ds_test = Dataset.from_pandas(test_df[["text","labels"]])


# ===========================================
# ||                                       ||
# ||Section 5: tokenization, tensorization ||
# ||              and collider             ||
# ||                                       ||
# ===========================================

# IMPORTING OUR TOKENIZER
tokz = AutoTokenizer.from_pretrained(model_nm)

# GETTING MAX LENGTH
max_length = get_max_lenghts(get_list_of_lengths(ds_train["text"], tokz))

# DEFINING A TOKENIZE FUNCTION TO TOKENIZE BOTH THE TWO DATASETS
def tok_func(x): return tokz(x["text"], truncation=True, padding = "max_length", max_length=max_length)

# CHECK THAT TOKENIZER FUNCTION WORKS
tok_func(ds_train[19]) # the 1 are for padding it; the attention mask show to not care about the 1

# TOKENIZING THE DS
tok_ds_train = ds_train.map(tok_func, batched=True, remove_columns=['text'])
tok_ds_validation = ds_validation.map(tok_func, batched=True, remove_columns=['text'])
tok_ds_test = ds_test.map(tok_func, batched=True, remove_columns=['text'])

# Use the `map()` method to apply the `convert_to_torch()` function to each item in the dataset
tok_ds_train = tok_ds_train.map(convert_to_torch, batched=True)

# Convert the dataset to a PyTorch TensorDataset by simply transforming each numerical column of the dataset in tensor
tensor_train_dataset = torch.utils.data.TensorDataset(torch.tensor(tok_ds_train['input_ids']),
                                                torch.tensor(tok_ds_train['token_type_ids']),
                                                torch.tensor(tok_ds_train['attention_mask']),
                                                torch.tensor(tok_ds_train['label']))
# SAME FOR VALIDATION SET

# Use the `map()` method to apply the `convert_to_torch()` function to each item in the dataset
tok_ds_validation = tok_ds_validation.map(convert_to_torch, batched=True)

# Convert the dataset to a PyTorch TensorDataset
tensor_validation_dataset = torch.utils.data.TensorDataset(torch.tensor(tok_ds_validation['input_ids']),
                                                torch.tensor(tok_ds_validation['token_type_ids']),
                                                torch.tensor(tok_ds_validation['attention_mask']),
                                                torch.tensor(tok_ds_validation['label']))

# SAME FOR TEST SET

# Use the `map()` method to apply the `convert_to_torch()` function to each item in the dataset
tok_ds_test = tok_ds_test.map(convert_to_torch, batched=True)

# Convert the dataset to a PyTorch TensorDataset
tensor_test_dataset = torch.utils.data.TensorDataset(torch.tensor(tok_ds_test['input_ids']),
                                                torch.tensor(tok_ds_test['token_type_ids']),
                                                torch.tensor(tok_ds_test['attention_mask']),
                                                torch.tensor(tok_ds_test['label']))


# PLUGGING INTO DATALOADERS

train_dataloader = DataLoader(
            tensor_train_dataset,  # The training samples.
            sampler = RandomSampler(tensor_train_dataset), # Select batches randomly
            batch_size = 32 # Trains with this batch size.
        )

validation_dataloader = DataLoader(
            tensor_validation_dataset,  # The training samples.
            sampler = RandomSampler(tensor_validation_dataset), # Select batches randomly
            batch_size = 32 # Trains with this batch size.
        )

test_dataloader = DataLoader(
            tensor_test_dataset,  # The training samples.
            sampler = RandomSampler(tensor_test_dataset), # Select batches randomly
            batch_size = 32 # Trains with this batch size.
        )

# ===========================================
# ||                                       ||
# ||Section 6: building the model          ||
# ||                                       ||
# ===========================================

# creating a pytorch module => that is a block of parameters and computation (forward)
class Bert4BinaryClassification(nn.Module):

    # initiliazer, specify the name of the bert model you want to load
    def __init__(self, model_name):

        # be sure the nn.Module is correctly
        super().__init__()

        # initialize the model (think to it as a cooler sequential(...))
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(getting_device())

    # forward method, we need to feed it with the tokenized text (ids + attention mask)
    def forward(self, input_ids, attention_mask):

        # pass the tokenized test through the model, which has as last layer a FNN with 2 output perceptrons
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # gather the 2 entries output vector
        logits = output.logits

        # return it
        return logits

    # implement the gpu util function as a method so it move directly the model on the gpu
    def getting_device(self, gpu_prefence=True):
        """
        This function gets the torch device to be used for computations,
        based on the GPU preference specified by the user.
        """

        # If GPU is preferred and available, set device to CUDA
        if gpu_prefence and torch.cuda.is_available():
            device = torch.device('cuda')
        # If GPU is not preferred or not available, set device to CPU
        else:
            device = torch.device("cpu")

        # Print the selected device
        print(f"Selected device for BERTBINARYCLASSIFICATION: {device}")

        # Return the device
        return device


# INITLIAZING THE MODEL AND CHECKING IF ON GPU
model = Bert4BinaryClassification(model_nm)
# checking if the model is on the gpu
print_gpu_utilization()


# ===========================================
# ||                                       ||
# ||Section 7: loss, optimizer and         ||
# ||              hyperparameters          ||
# ||                                       ||
# ===========================================

# 1) LOSS
loss_fn = nn.CrossEntropyLoss() # input => (predicted probab positive class, positive or negative class)


# 2) OPTIMIZER

# hyperparameters 1: optmizer

learning_rate_hp1 = 6e-6 # standard

epsilon_value_hp2 = 1e-8 # standard

# setting up the optmizer

# it is a transformer class, it requires two argument, a.k.a the parameters it needs to update at each step and the learning rate to scale the gradient

optimizer1 = AdamW(model.parameters(),
                  lr = learning_rate_hp1, # args.learning_rate
                  eps = epsilon_value_hp2 # args.adam_epsilon
                )

# 3) LR SCHEDULER

# hyperparameters 2: lr_scheduler

epochs_hp3 = 3 # total number of epochs for training
total_steps = len(train_dataloader) * epochs_hp3 # Total number of training steps is the number of steps per epoch times the number of epochs
warmup_steps_hp4 = 0 # so just decay not increasing of learning rate, seems to be standard

# setting up the scheduler

scheduler = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps = warmup_steps_hp4,num_training_steps = total_steps)


# ===========================================
# ||                                       ||
# ||Section 8: training, testing, and      ||
# ||            validation functions       ||
# ||                                       ||
# ===========================================

# 1) TRAINING

def train(model, train_dataloader, loss_fn, optimizer, scheduler):
    """
    This function trains a given model on training data and validates on validation data
    """
    # setting the model to training mode => important because it says to compute gradients 4 backwards pass, while computing the forward pass
    model.train()

    # wrapping tqdm around the dataloader to allow visualization
    visual_train_dl = tqdm(train_dataloader)

    # initiliaze it to compute the training loss after all the batches
    total_train_loss = 0

    # initialize for printing near the bar
    train_step = 0

    # iterate over the batch of the data loader (after all iteration we made 1 epoch)
    for batch in visual_train_dl:
        # accessing batch (contains an input_ids, an attention mask, and a label)
        batch_ids = batch[0].to(device)
        batch_attention_mask = batch[2].to(device)
        # squeeze remove dimension because labels should not have dimension, long transform in long integer as required by pytorch
        batch_labels = batch[3].squeeze().to(device).long()

        # step 1: reset optimizer stored gradient
        optimizer.zero_grad()

        # step 2: model logists through forward pass => remember the forward return logists
        logits = model(batch_ids, batch_attention_mask).to(device)

        # step 3: take the argmax index aka the predicted class index
        predictions = torch.argmax(logits, dim=1)

        # detached_predictions = predictions.detach().cpu().numpy()

        # step 4: compute the loss => takes as input the logist aka the predicted probability, not the predicted class
        loss = loss_fn(logits, batch_labels)
        # needed for printing stats
        total_train_loss += loss.item()
        train_step += 1

        # step 5: compute the gradient (derivative of the loss over every trainable parameter)
        loss.backward()

        # step 6: advance the optimizer and the scheduler
        optimizer.step()
        scheduler.step()

        # The set_postfix() method of the progress bar updates the message displayed in the progress bar to include the specified key-value pairs
        visual_train_dl.set_postfix({'train_loss': total_train_loss / train_step})
    visual_train_dl.close()

    # Calculate the average loss over all of the batches.

    final_avg_train_loss = total_train_loss / train_step

    print('')
    print('  Average training loss: {0:.2f}'.format(final_avg_train_loss))


# 2) VALIDATION WITH ACCURACY

def validate(model, valid_dataloader, loss_fn):
    # step 1, say to the model that computing the forward is enough, no backward!
    with torch.no_grad():
        # step 2, say to the model that it is validation time (4 dropout and normalization)
        model.eval()

        # initiliaze it to compute the training loss after all the batches
        total_valid_loss = 0

        # initiliaze to compute the avg thereafter
        valid_step = 0

        # list for accuracy
        correct = 0

        # wrapper for progress bar
        visual_valid_dl = tqdm(valid_dataloader)

        # iterate over the batch of the data loader (after all iteration we made 1 epoch)
        for batch in visual_valid_dl:
            # update the step
            valid_step += 1

            # accessing batch (contains an input_ids, an attention mask, and a label)
            batch_ids = batch[0].to(device)
            batch_attention_mask = batch[2].to(device)

            # squeeze remove dimension because labels should not have dimension, long transform in long integer as required by pytorch
            batch_labels = batch[3].squeeze().to(device).long()

            # step 3: model logists through forward pass => remember the forward return logists
            logits = model(batch_ids, batch_attention_mask).to(device)

            # step 4: getting predictions
            predictions = torch.argmax(logits, dim=1)

            # step 5: check if correct
            correct += (predictions == batch_labels).type(torch.float).sum().item()

            # detached_predictions = predictions.detach().cpu().numpy()

            # step 5: computing the loss
            loss = loss_fn(logits, batch_labels)

            # step 6: add to total loss
            total_valid_loss += loss.item()

        total_valid_loss /= valid_step
        accuracy = correct / len(valid_dataloader.dataset)

        print(f'Accuracy Score: {accuracy}')
        print(f'Valid_loss: {total_valid_loss}')


# 3) TESTING WITH ACCURACY AND F1

def test_with_f1(model, test_dataloader, loss_fn):
    # step 1, say to the model that computing the forward is enough, no backward!
    with torch.no_grad():
        # step 2, say to the model that it is validation time (4 dropout and normalization)
        model.eval()

        # initiliaze it to compute the training loss after all the batches
        total_test_loss = 0

        # initiliaze to compute the avg thereafter
        test_step = 0

        # list for accuracy
        correct = 0

        # list for predictions and true labels
        all_predictions = []
        all_labels = []

        # wrapper for progress bar
        visual_test_dl = tqdm(test_dataloader)

        # iterate over the batch of the data loader (after all iteration we made 1 epoch)
        for batch in visual_test_dl:
            # update the step
            test_step += 1

            # accessing batch (contains an input_ids, an attention mask, and a label)
            batch_ids = batch[0].to(device)
            batch_attention_mask = batch[2].to(device)

            # squeeze remove dimension because labels should not have dimension, long transform in long integer as required by pytorch
            batch_labels = batch[3].squeeze().to(device).long()

            # step 3: model logists through forward pass => remember the forward return logists
            logits = model(batch_ids, batch_attention_mask).to(device)

            # step 4: getting predictions
            predictions = torch.argmax(logits, dim=1)

            # append predictions and labels to lists
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(batch_labels.cpu().numpy().tolist())

            # step 5: check if correct
            correct += (predictions == batch_labels).type(torch.float).sum().item()

            # detached_predictions = predictions.detach().cpu().numpy()

            # step 5: computing the loss
            loss = loss_fn(logits, batch_labels)

            # step 6: add to total loss
            total_test_loss += loss.item()

        total_test_loss /= test_step
        accuracy = correct / len(test_dataloader.dataset)
        f1 = f1_score(all_labels, all_predictions, average='macro')

        print(f'Accuracy Score: {accuracy}')
        print(f'F1 Score: {f1}')
        print(f'Test_loss: {total_test_loss}')

# ===========================================
# ||                                       ||
# ||Section 9: training and testing        ||
# ||                                       ||
# ===========================================

# TRAINING LOOP
print(" ")
print("START TRAINING ")
print(" ")
for t in range(epochs_hp3):
    print(f"Epoch {t+1}\n-------------------------------")
    train(model = model, train_dataloader = train_dataloader, loss_fn = loss_fn, optimizer = optimizer1, scheduler = scheduler)
    validate(model = model,valid_dataloader = validation_dataloader,loss_fn = loss_fn)
print("DONE TRAINING")

# TESTING
print(" ")
print("START TESTING")
print(" ")
test_with_f1(model = model,valid_dataloader = test_dataloader,loss_fn = loss_fn)
print("DONE TESTING")


# ===========================================
# ||                                       ||
# ||Section 10: saving the model           ||
# ||                                       ||
# ===========================================

model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'BERT_weights.pth')