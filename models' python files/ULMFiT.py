!pip install optuna

# ===========================================
# ||                                       ||
# ||       Importing needed libraries      ||
# ||                                       ||
# ===========================================

from fastai.text.all import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import optuna

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


# Read the CSV file into a variable (dataset from https://www.kaggle.com/datasets/vstepanenko/disaster-tweets)
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
# ||       Section 2: hyperparameter       ||
# ||                        tuning         ||
# ||                                       ||
# ===========================================


def train_language_model(trial):

    print(" NUOVO ROUND LM HYP. SEARCH")
    # Define the hyperparameters to optimize
    wheight_decay_HP1 = trial.suggest_float('wheight_decay_HP1', 1e-5, 1e-1, log=True)
    learningrate_1_HP2 = trial.suggest_float('learningrate_1_HP2', 1e-5, 1e-1, log=True)
    learningrate_2_HP3 = trial.suggest_float('learningrate_2_HP3', 1e-5, 1e-1, log=True)
    epoch1_HP4 = trial.suggest_int('epoch1_HP4', 1, 10)
    epoch2_HP5 = trial.suggest_int('epoch2_HP5', 1, 10)

    # Create a learner object
    learn = language_model_learner(dls_lm, AWD_LSTM, metrics=[accuracy, Perplexity()], wd=wheight_decay_HP1).to_fp16()

    # Fine-tune the language model for one epoch
    learn.fit_one_cycle(n_epoch=epoch1_HP4, lr_max=learningrate_1_HP2)

    # Unfreeze the layers of the language model and fine-tune it
    learn.unfreeze()
    learn.fit_one_cycle(n_epoch=epoch2_HP5, lr_max=learningrate_2_HP3)

    # Get the validation loss after fine-tuning
    val_loss = learn.validate()[0]

    return val_loss

# Run the hyperparameter optimization for the language model
study_lm = optuna.create_study(direction='minimize')
study_lm.optimize(train_language_model, n_trials=50)

# Get the best hyperparameters
best_wheight_decay_HP1 = study_lm.best_params['wheight_decay_HP1']
best_learningrate_1_HP2 = study_lm.best_params['learningrate_1_HP2']
best_learningrate_2_HP3 = study_lm.best_params['learningrate_2_HP3']
best_epoch_1_HP4 = study_lm.best_params['epoch1_HP4']
best_epoch_2_HP5 = study_lm.best_params['epoch2_HP5']


# ===========================================
# ||                                       ||
# ||       Section 3: train the language   ||
# ||                            model      ||
# ||                                       ||
# ===========================================


# Create a learner object
learn = language_model_learner(dls_lm, AWD_LSTM, metrics=[accuracy, Perplexity()], wd=best_wheight_decay_HP1).to_fp16()

# Fine-tune the language model for one epoch
learn.fit_one_cycle(n_epoch = best_epoch_1_HP4, lr_max= best_learningrate_1_HP2)

# Unfreeze the layers of the language model and fine-tune it
learn.unfreeze()
learn.fit_one_cycle(n_epoch = best_epoch_2_HP5, lr_max= best_learningrate_2_HP3)

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

# Create a dataloader
dls_clas = TextDataLoaders.from_df(train_df, valid_df=validation_df, path='.', valid_pct=0.2, seed=None,
                          text_col=0, label_col=1, label_delim=None,
                          y_block=None, text_vocab=dls_lm.vocab, is_lm=False,
                          valid_col=None, tok_tfm=None,
                          tok_text_col='text', seq_len=72)


# ===========================================
# ||                                       ||
# ||       Section 2: hyperparameter       ||
# ||                        tuning         ||
# ||                                       ||
# ===========================================


def train_classifier_model(trial):

    print(" NUOVO ROUND class HYP. SEARCH")

    # Define the hyperparameters as Optuna parameters
    lr_1_hp5 = trial.suggest_loguniform('lr_1_hp5', 1e-4, 1e-1)
    lr_2_hp6 = trial.suggest_categorical('lr_2_hp6', [slice(1e-2/(2.6**4),1e-2), slice(1e-3/(2.6**4),1e-3)])
    lr_3_hp7 = trial.suggest_categorical('lr_3_hp7', [slice(5e-3/(2.6**4),5e-3), slice(1e-3/(2.6**4),1e-3)])
    lr_4_hp8 = trial.suggest_loguniform('lr_4_hp8', 1e-7, 1e-3)
    drop_mult_hp9 = trial.suggest_uniform('drop_mult_hp9', 0.1, 0.9)

    # Create a learner object for training a text classifier
    learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=drop_mult_hp9, metrics=accuracy)

    # Load the encoder part of the previously fine-tuned language model to the classifier learner object
    learn = learn.load_encoder('finetuned')

    # Train the classifier for one epoch
    learn.fit_one_cycle(1, lr_1_hp5)

    # Freeze all but the last two layers of the classifier and fine-tune it for one epoch
    learn.freeze_to(-2)
    learn.fit_one_cycle(1, lr_2_hp6)

    # Freeze all but the last three layers of the classifier and fine-tune it for one epoch
    learn.freeze_to(-3)
    learn.fit_one_cycle(1, lr_3_hp7)

    # Unfreeze all layers of the classifier and fine-tune it for three epochs
    learn.unfreeze()
    learn.fine_tune(2, lr_4_hp8, cbs=[ShowGraphCallback()])

        # Get the predicted probabilities for the validation data using the trained model.
    val_dl = dls_clas.test_dl(validation_df['text'])
    val_preds, _ = learn.get_preds(dl=val_dl)

    # Get the predicted labels for the validation data.
    val_predicted_labels = val_preds.argmax(dim=1)

    # Compute the f1 score of the model on the validation data using the `f1_score` function.
    val_f1 = f1_score(validation_df["target"].values, val_predicted_labels)

    # Return the negative f1 score as the loss to optimize (because optuna maximizes the negative of the objective).
    return -val_f1

# Create an optuna study and optimize the objective function using the TPE sampler.
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
study.optimize(train_classifier_model, n_trials=2)

# Print the best set of hyperparameters found by optuna and the corresponding f1 score on the validation data.
print('Best trial:')
best_trial = study.best_trial
print(f'  Value: {-best_trial.value:.5f}')
print('  Params: ')
for key, value in best_trial.params.items():
    print(f'    {key}: {value}')

# Hyperparameters

best_lr_1_hp5 = best_trial.params['lr_1_hp5']
best_lr_2_hp6 = best_trial.params['lr_2_hp6']
best_lr_3_hp7 = best_trial.params['lr_3_hp7']
best_lr_4_hp8 = best_trial.params['lr_4_hp8']
best_drop_mult_hp9 = best_trial.params['drop_mult_hp9']


# ===========================================
# ||                                       ||
# ||       Section 3: train the model      ||
# ||                                       ||
# ===========================================


# Create a learner object for training a text classifier
learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=best_drop_mult_hp9, metrics=accuracy)

# Load the encoder part of the previously fine-tuned language model to the classifier learner object
learn = learn.load_encoder('finetuned')



# Train the classifier for one epoch
learn.fit_one_cycle(1, best_lr_1_hp5)

# Freeze all but the last two layers of the classifier and fine-tune it for one epoch
learn.freeze_to(-2)
learn.fit_one_cycle(1, best_lr_2_hp6)

# Freeze all but the last three layers of the classifier and fine-tune it for one epoch
learn.freeze_to(-3)
learn.fit_one_cycle(1, best_lr_3_hp7)

# Unfreeze all layers of the classifier and fine-tune it for three epochs
learn.unfreeze()
learn.fine_tune(2, best_lr_4_hp8, cbs=[ShowGraphCallback()])

# ===========================================
# ||                                       ||
# ||       Section 4: test the model       ||
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

# ===========================================
# ||                                       ||
# ||       Section 5: Save the model       ||
# ||                                       ||
# ===========================================

learn.export("ULMFiT.pkl")