# ===========================================
# ||                                       ||
# ||       Section 1: Importing modules    ||
# ||                                       ||
# ===========================================

from fastai.text.all import *
import pandas as pd
from sklearn.model_selection import train_test_split

# ===========================================
# ||                                       ||
# ||       Section 2: getting dataframes   ||
# ||                    and dataloader     ||
# ||                                       ||
# ===========================================

#TODO use clean dataset
# Read in a CSV file named "train.csv" from a specific file path
# and store it in a pandas dataframe named "df".
df = pd.read_csv("/content/drive/MyDrive/ML_proj/train.csv")

# Remove the "id", "keyword", and "location" columns from the dataframe "df".
df.drop(columns=["id", "keyword", "location"], inplace=True)

# Split the dataframe into train and test dataframes
train_df, test_df = train_test_split(df, test_size=0.2, random_state=69)

# Create a data loader for text data using the "TextDataLoaders" class from the fastai library.
dls = TextDataLoaders.from_df(train_df, path='.', valid_pct=0.2, seed=None,
                          text_col=0, label_col=1, label_delim=None,
                          y_block=None, text_vocab=None, is_lm=False,
                          valid_col=None, tok_tfm=None,
                          tok_text_col='text', seq_len=72)

# ===========================================
# ||                                       ||
# ||       Section 3: train the model      ||
# ||                                       ||
# ===========================================

# Picking hyperparameters
dropout_hp1 = 0.5
epoch_hp2 = 4
lr_hp3 = 1e-2

# Create a text classification learner using the fastai library.
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=dropout_hp1, metrics=accuracy)

# Fine-tune the neural network for four epochs using stochastic gradient descent
learn.fine_tune(epoch_hp2, lr_hp3)

# ===========================================
# ||                                       ||
# ||       Section 4: testing the model    ||
# ||                                       ||
# ===========================================

# Create a data loader for the test data using the same parameters as the training data loader.
test_dl = dls.test_dl(test_df['text'])

# Get the predicted labels and probabilities for the test data using the trained model.
preds, targets = learn.get_preds(dl=test_dl, with_decoded=True)

# Print the accuracy of the model on the test data.
acc = accuracy(preds, targets)
print(f"Test accuracy: {acc}")