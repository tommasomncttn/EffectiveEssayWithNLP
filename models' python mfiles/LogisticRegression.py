# ===========================================
# ||                                       ||
# ||       Section 1: Importing modules    ||
# ||                                       ||
# ===========================================

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split

# ===========================================
# ||                                       ||
# ||       Section 2: getting dataframes   ||
# ||                                       ||
# ===========================================

# define path2file
path2file = "C:/Users/luigi/OneDrive/Desktop/progetto_ML/train.csv"
# Load the dataset
df = pd.read_csv(path2file)

# Splitting the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=69)

# Extracting the text and target labels from the training and testing data
text_train = train_df["text"]
label_train = train_df["target"].values
text_test = test_df["text"]
label_test = test_df["target"].values

# ===========================================
# ||                                       ||
# ||       Section 3: extract numeric      ||
# ||                  features             ||
# ||                                       ||
# ===========================================

# Create CountVectorizer object
vect = CountVectorizer(stop_words='english')

# Fit the CountVectorizer on the training data to learn the vocabulary and create a document-term matrix
vectorized_train = vect.fit_transform(text_train)

# Transforming the test data into the same document-term matrix as the training data
vectorized_test = vect.transform(text_test)

# ===========================================
# ||                                       ||
# ||       Section 3: train the model      ||
# ||                                       ||
# ===========================================

# Set number of folds to use for cross-validation
num_folds = 5

# Create a KFold object to split the data into K folds
kf = KFold(n_splits=num_folds, shuffle=True)

# Create an empty list to store F1 scores for each fold
f1_scores = []

# Loop over each fold and train the model on the training data, then evaluate on the validation data
for fold, (train_indices, val_indices) in enumerate(kf.split(vectorized_train, label_train)):

    # Split the data into training and validation sets for this fold
    X_train, y_train = vectorized_train[train_indices], label_train[train_indices]
    X_val, y_val = vectorized_train[val_indices], label_train[val_indices]

    # Create logistic regression model
    model = LogisticRegression(max_iter=500)

    # Hyperparameter search
    # param_grid = {'C': np.logspace(-3, 3, 7), 'penalty': ['l2', None]}
    # model_H = GridSearchCV(model,param_grid,cv=3)

    # Fit a logistic regression model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the validation data using the trained model
    predicted_labels = model.predict(X_val)

    # Evaluate the performance of the model using F1 score
    f1 = f1_score(predicted_labels, y_val)

    # Add the F1 score for this fold to the list
    f1_scores.append(f1)

    # Print the F1 score for this fold
    print(f"Fold {fold}: F1 score = {f1}")

    # Calculate the average F1 score across all folds
    avg_score = np.mean(f1_scores)
    print(f"Average F1 score across {num_folds} folds: {avg_score}")

# ===========================================
# ||                                       ||
# ||       Section 4: test the model      ||
# ||                                       ||
# ===========================================

# Create a new logistic regression model
model = LogisticRegression(max_iter=500)

# Fit a logistic regression model on the full training data
model.fit(vectorized_train, label_train)

# Make predictions on the validation data using the trained model
predicted_labels = model.predict(vectorized_test)

# Evaluate the performance of the model using F1 score
f1 = f1_score(predicted_labels, label_test)

# Print the performance of the final model
print(f"F1 score for the model trained on all the training data: {f1}")