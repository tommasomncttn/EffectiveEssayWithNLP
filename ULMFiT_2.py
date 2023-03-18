# ===========================================
# ||                                       ||
# ||       Importing modules               ||
# ||                                       ||
# ===========================================

from fastai.text.all import *
import pandas as pd
from sklearn.model_selection import train_test_split


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

# Import the pandas library and read the CSV file into a variable called "dataset"
dataset = pd.read_csv("/content/drive/MyDrive/ML_proj/tweets.csv",)

# Drop the "id", "keyword", "location", and "target" columns from the dataset in-place
dataset.drop(columns = ["id", "keyword", "location", "target"], inplace =True)

# Display the first five rows of the modified dataset
dataset.head()

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

l# Use the language_model_learner function from the fastai library to create a learner object for fine-tuning
# a language model using the data loader created earlier (dls_lm), an AWD_LSTM model architecture,
# and a list of metrics (accuracy and Perplexity). Set the weight decay to 0.1 and convert the model to fp16.
learn = language_model_learner(dls_lm, AWD_LSTM, metrics=[accuracy, Perplexity()], wd=0.1).to_fp16()

# Fine-tune the language model for one epoch using a learning rate of 1e-2
learn.fit_one_cycle(1, 1e-2)

# Unfreeze the layers of the language model and fine-tune it for 10 epochs using a smaller learning rate of 1e-3
learn.unfreeze()
learn.fit_one_cycle(10, 1e-3)

# Save the encoder part of the fine-tuned language model to a file named "finetuned"
learn.save_encoder('finetuned')

# Set the initial text prompt and the desired number of generated words and sentences
TEXT = "there was a fire"
N_WORDS = 40
N_SENTENCES = 2

# Generate text using the fine-tuned language model by calling the predict method of the learner object,
# passing in the text prompt, the number of words to generate, and the temperature (0.75) as arguments.
# Repeat the generation process N_SENTENCES times and store the results in a list called preds.
preds = [learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)]

# Print the generated text by joining the elements of the preds list with a newline character between them
print("\n".join(preds))



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


# Use the TextDataLoaders class from the fastai library to create a dataloader for language modeling
# using the modified dataset as the source data
dls_clas = TextDataLoaders.from_df(df, path='.', valid_pct=0.2, seed=None,
                          text_col=0, label_col=1, label_delim=None,
                          y_block=None, text_vocab=dls_lm.vocab, is_lm=False,
                          valid_col=None, tok_tfm=None,
                          tok_text_col='text', seq_len=72)



# ===========================================
# ||                                       ||
# ||       Section 2: train the model      ||
# ||                                       ||
# ===========================================


# Use the text_classifier_learner function from the fastai library to create a learner object for training
# a text classifier using the data loader created earlier (dls_clas), an AWD_LSTM model architecture,
# a dropout multiplier of 0.5, and a list of metrics (accuracy).
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
learn.fit_one_cycle(4, slice(1e-3/(2.6**4),1


