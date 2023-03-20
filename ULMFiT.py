# ===========================================
# ||                                       ||
# ||       Importing modules               ||
# ||                                       ||
# ===========================================

from fastai.text.all import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ===========================================
# ||                                       ||
# ||       PART 1: Language Model          ||
# ||                                       ||
# ===========================================
# ===========================================
# ||                                       ||
# ||       Section 1: getting dataframes   ||
# ||                    and dataloader     ||
# ||                                       ||
# ===========================================


# Read the CSV file into a variable
dataset = pd.read_csv("/content/drive/MyDrive/ML_proj/tweets.csv",)

# Drop the "id", "keyword", "location", and "target" columns from the dataset
dataset.drop(columns = ["id", "keyword", "location", "target"], inplace =True)

# Create a dataloader for language modeling
dls_lm = TextDataLoaders.from_df(dataset, path='.', valid_pct=0.2, seed=None,
                          text_col=0, label_col=1, label_delim=None,
                          y_block=None, text_vocab=None, is_lm=True,
                          valid_col=None, tok_tfm=None,
                          tok_text_col='text', seq_len=72, bs=32)


# ===========================================
# ||                                       ||
# ||       Section 2: train the language   ||
# ||                            model      ||
# ||                                       ||
# ===========================================

wheight_decay_HP1 = 0.1
learningrate_1_HP2 = 0.00363078061491251
learningrate_2_HP3 = 1e-3
epoch1_HP4 = 2
epoch2_hp5 = 6
# Create a learner object
learn = language_model_learner(dls_lm, AWD_LSTM, metrics=[accuracy, Perplexity()], wd=wheight_decay_HP1).to_fp16()
print(learn.lr_find())
# Fine-tune the language model for one epoch
learn.fit_one_cycle(n_epoch = epoch1_HP4, lr_max= learningrate_1_HP2)

# Unfreeze the layers of the language model and fine-tune it
learn.unfreeze()
learn.fit_one_cycle(n_epoch = epoch2_hp5, lr_max= learningrate_2_HP3)

# Save the encoder part of the fine-tuned language model
learn.save_encoder('finetuned')

# ===========================================
# ||                                       ||
# ||       PART 2: Classifier              ||
# ||                                       ||
# ===========================================
# ===========================================
# ||                                       ||
# ||       Section 1: getting dataframes   ||
# ||                    and dataloader     ||
# ||                                       ||
# ===========================================

#TODO use clean dataset
# Read in a CSV file
train_df = pd.read_csv("/content/drive/MyDrive/ML_proj/zaazazza/train_df.csv")
test_df = pd.read_csv("/content/drive/MyDrive/ML_proj/zaazazza/test_df.csv")
validation_df =  pd.read_csv("/content/drive/MyDrive/ML_proj/zaazazza/validation_df.csv")

# Drop not needed columns
train_df = test_df.drop(train_df.columns[0:4], axis=1)
validation_df = validation_df.drop(validation_df.columns[0:4], axis=1)
test_df = test_df.drop(test_df.columns[0:4], axis=1)
print(test_df.columns)
# Create a dataloader
dls_clas = TextDataLoaders.from_df(train_df, valid_df=validation_df, path='.', valid_pct=0.2, seed=None,
                          text_col=0, label_col=1, label_delim=None,
                          y_block=None, text_vocab=dls_lm.vocab, is_lm=False,
                          valid_col=None, tok_tfm=None,
                          tok_text_col='text', seq_len=72)

# ===========================================
# ||                                       ||
# ||       Section 2: train the model      ||
# ||                                       ||
# ===========================================

# Create a learner object for training a text classifier
learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.7, metrics=accuracy)

# Load the encoder part of the previously fine-tuned language model to the classifier learner object
learn = learn.load_encoder('finetuned')

# Train the classifier for one epoch
#learn.fit_one_cycle(1, 2e-2)

# Freeze all but the last two layers of the classifier and fine-tune it for one epoch
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2))

# Freeze all but the last three layers of the classifier and fine-tune it for one epoch
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))

# Unfreeze all layers of the classifier and fine-tune it for three epochs
learn.unfreeze()
learn.fine_tune(2, 1e-5,1, cbs=[ShowGraphCallback()])

# ===========================================
# ||                                       ||
# ||       Section 3: test the model       ||
# ||                                       ||
# ===========================================

# Create a test dataloader from the test data using the `test_dl` method of the `dls` dataloaders object.
test_dl = dls_clas.test_dl(test_df['text'])

# Get the predicted probabilities for the test data using the trained model.
preds, _ = learn.get_preds(dl=test_dl)

# Get the predicted labels for the test data.
predicted_labels = preds.argmax(dim=1)

# Convert the predicted labels to Python list and get the corresponding class names.
predicted_classes = [dls_clas.vocab[i] for i in predicted_labels]

# Convert the predicted classes list to a tensor.
predicted_classes_tensor = torch.tensor(predicted_labels)

# Reshape the predicted tensor to have the same shape as the target tensor.
predicted_classes_tensor = predicted_classes_tensor.unsqueeze(1)

# Convert the target labels to a tensor.
target_tensor = torch.tensor(test_df["target"].values)

# Compute the accuracy and f1 score of the model on the test data using the `accuracy` and `f1_score` functions.
acc = accuracy(predicted_classes_tensor, target_tensor)
f1 = f1_score(target_tensor, predicted_classes_tensor)

# Print the accuracy and f1 score of the model on the test data.
print(f"Test accuracy: {acc}")
print(f"Test f1 score: {f1}")