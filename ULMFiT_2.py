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


# Read the CSV file into a variable called "dataset"
dataset = pd.read_csv("/content/drive/MyDrive/ML_proj/tweets.csv",)

# Drop the "id", "keyword", "location", and "target" columns from the dataset in-place
dataset.drop(columns = ["id", "keyword", "location", "target"], inplace =True)

# Use the TextDataLoaders class from the fastai library to create a dataloader for language modeling
# using the modified dataset as the source data
dls_lm = TextDataLoaders.from_df(dataset, path='.', valid_pct=0.2, seed=None,
                          text_col=0, label_col=1, label_delim=None,
                          y_block=None, text_vocab=None, is_lm=True,
                          valid_col=None, tok_tfm=None,
                          tok_text_col='text', seq_len=72, bs=32)



# ===========================================
# ||                                       ||
# ||       Section 2: train the model      ||
# ||                                       ||
# ===========================================

# Use the language_model_learner function from the fastai library to create a learner object
learn = language_model_learner(dls_lm, AWD_LSTM, metrics=[accuracy, Perplexity()], wd=0.1).to_fp16()

# Fine-tune the language model for one epoch using a learning rate of 1e-2
learn.fit_one_cycle(1, 1e-2)

# Unfreeze the layers of the language model and fine-tune it for 10 epochs using a smaller learning rate of 1e-3
learn.unfreeze()
learn.fit_one_cycle(10, 1e-3)

# Save the encoder part of the fine-tuned language model to a file named "finetuned"
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
# Read in a CSV file named "train.csv" from a specific file path
# and store it in a pandas dataframe named "df".
df = pd.read_csv("/content/drive/MyDrive/ML_proj/train.csv")

# Remove the "id", "keyword", and "location" columns from the dataframe "df".
df.drop(columns=["id", "keyword", "location"], inplace=True)

# Split the dataframe into train and test dataframes
train_df, test_df = train_test_split(df, test_size=0.2, random_state=69)

# Use the TextDataLoaders class from the fastai library to create a dataloader for language modeling
# using the modified dataset as the source data
dls_clas = TextDataLoaders.from_df(train_df, path='.', valid_pct=0.2, seed=None,
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
learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, metrics=accuracy)

# Load the encoder part of the previously fine-tuned language model to the classifier learner object
learn = learn.load_encoder('finetuned')

# Train the classifier for one epoch using a learning rate of 2e-2
learn.fit_one_cycle(1, 2e-2)

# Freeze all but the last two layers of the classifier and fine-tune it for one epoch using a slice of
# the learning rate (1e-2/(2.6**4), 1e-2) as the learning rate schedule
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2))

# Freeze all but the last three layers of the classifier and fine-tune it for one epoch using a slice of
# the learning rate (5e-3/(2.6**4), 5e-3) as the learning rate schedule
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))

# Unfreeze all layers of the classifier and fine-tune it for four epochs using a slice of the learning rate
# (1e-3/(2.6**4), 1e-3) as the learning rate schedule
learn.unfreeze()
learn.fit_one_cycle(4, slice(1e-3/(2.6**4),1))

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