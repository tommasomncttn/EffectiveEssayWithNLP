# ===========================================
# ||                                       ||
# ||       Section 1: Importing modules    ||
# ||                                       ||
# ===========================================

from fastai.text.all import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

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
dropout_hp1 = 0.7
epoch_hp2 = 4
lr_hp3 = 1e-2

# Create a text classification learner using the fastai library.
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=dropout_hp1, metrics=accuracy)

# Fine-tune the neural network for four epochs using stochastic gradient descent
learn.fine_tune(epoch_hp2, lr_hp3, cbs=[ShowGraphCallback()])

# ===========================================
# ||                                       ||
# ||       Section 4: testing the model    ||
# ||                                       ||
# ===========================================

# Create a test dataloader from the test data using the `test_dl` method of the `dls` dataloaders object.
test_dl = dls.test_dl(test_df['text'])

# Get the predicted probabilities for the test data using the trained model.
preds, _ = learn.get_preds(dl=test_dl)

# Get the predicted labels for the test data.
predicted_labels = preds.argmax(dim=1)

# Convert the predicted labels to Python list and get the corresponding class names.
predicted_classes = [dls.vocab[i] for i in predicted_labels]

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