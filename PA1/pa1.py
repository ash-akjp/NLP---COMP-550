# Code: Real vs. Fake Fact Classification for Chennai


# Import Statements
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
import string
from sklearn.feature_selection import SelectKBest, chi2
import contractions

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load data
real_facts = open('facts.txt').readlines()
fake_facts = open('fakes.txt').readlines()

# Combine and label the dataset
facts = [(fact, 1) for fact in real_facts] + [(fact, 0) for fact in fake_facts]
df = pd.DataFrame(facts, columns=['text', 'label'])

# Split the dataset into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# Preprocessing: Tokenization, stopword removal, and lemmatization/stemming
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Function for lemmatization preprocessing
def preprocess_lemmatization(text):
    tokens = word_tokenize(text.lower())  # Lowercase and Tokenization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

# Function for stemming preprocessing
def preprocess_stemming(text):
    tokens = word_tokenize(text.lower())  # Lowercase and Tokenization
    tokens = [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

####---------------------------------------------------- Experiment 1: Stemming vs. Lemmatization (Logistic Regression)-------------------------------#

# Apply Lemmatization
X_train_lemma = X_train.apply(preprocess_lemmatization)
X_test_lemma = X_test.apply(preprocess_lemmatization)

# Apply Stemming
X_train_stem = X_train.apply(preprocess_stemming)
X_test_stem = X_test.apply(preprocess_stemming)

# Feature extraction: TF-IDF for lemmatization and stemming data
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Using unigrams and bigrams

# Lemmatization-based vectors
X_train_vec_lemma = vectorizer.fit_transform(X_train_lemma)
X_test_vec_lemma = vectorizer.transform(X_test_lemma)

# Stemming-based vectors
X_train_vec_stem = vectorizer.fit_transform(X_train_stem)
X_test_vec_stem = vectorizer.transform(X_test_stem)

# Hyperparameter grid for Logistic Regression
param_grid_logreg = {'C': [0.1, 1, 10], 'solver': ['liblinear']}

# Stratified K-Fold Cross-Validation (using 5 folds)
strat_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Logistic Regression with Lemmatization
log_reg = LogisticRegression()
grid_search_logreg_lemma = GridSearchCV(log_reg, param_grid_logreg, cv=strat_kfold, scoring='accuracy')
grid_search_logreg_lemma.fit(X_train_vec_lemma, y_train)

# Best model for Lemmatization
best_log_reg_lemma = grid_search_logreg_lemma.best_estimator_
print("Best Logistic Regression Model (Lemmatization):", best_log_reg_lemma)

# Evaluation on test set (Lemmatization)
y_test_pred_lr_lemma = best_log_reg_lemma.predict(X_test_vec_lemma)
print("\nLogistic Regression (Lemmatization) Test Accuracy:", accuracy_score(y_test, y_test_pred_lr_lemma))
print(classification_report(y_test, y_test_pred_lr_lemma))

# Logistic Regression with Stemming
grid_search_logreg_stem = GridSearchCV(log_reg, param_grid_logreg, cv=strat_kfold, scoring='accuracy')
grid_search_logreg_stem.fit(X_train_vec_stem, y_train)

# Best model for Stemming
best_log_reg_stem = grid_search_logreg_stem.best_estimator_
print("\nBest Logistic Regression Model (Stemming):", best_log_reg_stem)

# Evaluation on test set (Stemming)
y_test_pred_lr_stem = best_log_reg_stem.predict(X_test_vec_stem)
print("\nLogistic Regression (Stemming) Test Accuracy:", accuracy_score(y_test, y_test_pred_lr_stem))
print(classification_report(y_test, y_test_pred_lr_stem))


#### ---------------------------------------------------- Experiment 2 : Lemmatization + SVM ------------------------------------------------------- #

# Apply Lemmatization
X_train_lemma = X_train.apply(preprocess_lemmatization)
X_test_lemma = X_test.apply(preprocess_lemmatization)

# Feature extraction: TF-IDF for lemmatization and stemming data
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Using unigrams and bigrams

# Lemmatization-based vectors
X_train_vec_lemma = vectorizer.fit_transform(X_train_lemma)
X_test_vec_lemma = vectorizer.transform(X_test_lemma)

# Use Lemmatization data from above
svm = LinearSVC()
param_grid_svm = {'C': [0.1, 1, 10]}
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=strat_kfold)
grid_search_svm.fit(X_train_vec_lemma, y_train)

# Best SVM model for Lemmatization
best_svm = grid_search_svm.best_estimator_
print("\nBest SVM Model (Lemmatization):", best_svm)

# Evaluation on test set
y_test_pred_svm = best_svm.predict(X_test_vec_lemma)
print("\nSVM (Lemmatization) Test Accuracy:", accuracy_score(y_test, y_test_pred_svm))
print(classification_report(y_test, y_test_pred_svm))

####---------------------------------- Experiment 2.1 : Lemmatization, without stopword removal + SVM ------------------------------------------#

# Function for lemmatization preprocessing but keeping stopwords
def preprocess_lemmatization_without_stopword_removal(text):
    tokens = word_tokenize(text.lower())  # Lowercase and Tokenization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]  # No stopword removal
    return ' '.join(tokens)

# Apply this new preprocessing to your dataset
X_train_lemma = X_train.apply(preprocess_lemmatization_without_stopword_removal)
X_test_lemma = X_test.apply(preprocess_lemmatization_without_stopword_removal)

# Feature extraction: TF-IDF for lemmatization and stemming data
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Using unigrams and bigrams

# Lemmatization-based vectors
X_train_vec_lemma = vectorizer.fit_transform(X_train_lemma)
X_test_vec_lemma = vectorizer.transform(X_test_lemma)

# Use Lemmatization data from above
svm = LinearSVC()
param_grid_svm = {'C': [0.1, 1, 10]}
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=strat_kfold)
grid_search_svm.fit(X_train_vec_lemma, y_train)

# Best SVM model for Lemmatization
best_svm = grid_search_svm.best_estimator_
print("\nBest SVM Model (Lemmatization):", best_svm)

# Evaluation on test set
y_test_pred_svm = best_svm.predict(X_test_vec_lemma)
print("\nSVM (Lemmatization) Test Accuracy:", accuracy_score(y_test, y_test_pred_svm))
print(classification_report(y_test, y_test_pred_svm))

####  -------------------------------------------------- Experiment 3: Lemmatization + Naive Bayes ----------------------------------------------------------- #

# Apply Lemmatization
X_train_lemma = X_train.apply(preprocess_lemmatization)
X_test_lemma = X_test.apply(preprocess_lemmatization)

# Feature extraction: TF-IDF for lemmatization and stemming data
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Using unigrams and bigrams

# Lemmatization-based vectors
X_train_vec_lemma = vectorizer.fit_transform(X_train_lemma)
X_test_vec_lemma = vectorizer.transform(X_test_lemma)

# Use Lemmatization data from above
nb = MultinomialNB()
param_grid_nb = {'alpha': [0.1, 0.5, 1, 2]}
grid_search_nb = GridSearchCV(nb, param_grid_nb, cv=strat_kfold)
grid_search_nb.fit(X_train_vec_lemma, y_train)

# Best Naive Bayes model for Lemmatization
best_nb = grid_search_nb.best_estimator_
print("\nBest Naive Bayes Model (Lemmatization):", best_nb)

# Evaluation on test set
y_test_pred_nb = best_nb.predict(X_test_vec_lemma)
print("\nNaive Bayes (Lemmatization) Test Accuracy:", accuracy_score(y_test, y_test_pred_nb))
print(classification_report(y_test, y_test_pred_nb))

#### ------------------------------------------- Experiment 3.1: Lemmatization, with punctuation removal, without stopword removal ------------------------------ #

# Remove punctuation
def remove_punctuation(text):
    return ''.join([char for char in text if char not in string.punctuation])

# Function for lemmatization preprocessing with punctuation removal but keeping stopwords
def preprocess_lemmatization_with_punctuation_removal(text):
    text = remove_punctuation(text)  # Remove punctuation
    tokens = word_tokenize(text.lower())  # Lowercase and Tokenization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]  # No stopword removal
    return ' '.join(tokens)

# Apply this new preprocessing to your dataset
X_train_lemma = X_train.apply(preprocess_lemmatization_with_punctuation_removal)
X_test_lemma = X_test.apply(preprocess_lemmatization_with_punctuation_removal)

# Feature extraction: TF-IDF for lemmatization and stemming data
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Using unigrams and bigrams

# Lemmatization-based vectors
X_train_vec_lemma = vectorizer.fit_transform(X_train_lemma)
X_test_vec_lemma = vectorizer.transform(X_test_lemma)

# Use Lemmatization data from above
nb = MultinomialNB()
param_grid_nb = {'alpha': [0.1, 0.5, 1, 2]}
grid_search_nb = GridSearchCV(nb, param_grid_nb, cv=strat_kfold)
grid_search_nb.fit(X_train_vec_lemma, y_train)

# Best Naive Bayes model for Lemmatization
best_nb = grid_search_nb.best_estimator_
print("\nBest Naive Bayes Model (Lemmatization):", best_nb)

# Evaluation on test set
y_test_pred_nb = best_nb.predict(X_test_vec_lemma)
print("\nNaive Bayes (Lemmatization) Test Accuracy:", accuracy_score(y_test, y_test_pred_nb))
print(classification_report(y_test, y_test_pred_nb))

#### ---------------------------------------- Experiment 4: Additional preprocessing steps ---------------------------------------------------------#

#pip install contractions

# Remove punctuation
def remove_punctuation(text):
    return ''.join([char for char in text if char not in string.punctuation])

# Expand contractions
def expand_contractions(text):
    return contractions.fix(text)

# Function for lemmatization preprocessing with additional steps
def preprocess_lemmatization_additional_steps(text):
    text = expand_contractions(text)  # Expand contractions
    text = remove_punctuation(text)  # Remove punctuation
    tokens = word_tokenize(text.lower())  # Lowercase and Tokenization
    # tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words] # stop_word removal
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]  # No stopword removal
    return ' '.join(tokens)

# Apply the enhanced preprocessing function
X_train_lemma = X_train.apply(preprocess_lemmatization_additional_steps)
X_test_lemma = X_test.apply(preprocess_lemmatization_additional_steps)

# Feature extraction: TF-IDF for lemmatization data
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Using unigrams and bigrams
X_train_vec_lemma = vectorizer.fit_transform(X_train_lemma)
X_test_vec_lemma = vectorizer.transform(X_test_lemma)

# Hyperparameter grid for Logistic Regression
param_grid_logreg = {'C': [0.1, 1, 10]}
# Stratified K-Fold Cross-Validation (using 5 folds)
strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression with Lemmatization and extra preprocessing
log_reg = LogisticRegression()
grid_search_logreg_lemma = GridSearchCV(log_reg, param_grid_logreg, cv=strat_kfold)
grid_search_logreg_lemma.fit(X_train_vec_lemma, y_train)

# Best model for Lemmatization with extra preprocessing
best_log_reg_lemma = grid_search_logreg_lemma.best_estimator_
print("Best Logistic Regression Model (Lemmatization + Extra Preprocessing):", best_log_reg_lemma)

# Evaluation on test set (Lemmatization + Extra Preprocessing)
y_test_pred_lr_lemma = best_log_reg_lemma.predict(X_test_vec_lemma)
print("\nLogistic Regression (Lemmatization + Extra Preprocessing) Test Accuracy:", accuracy_score(y_test, y_test_pred_lr_lemma))
print(classification_report(y_test, y_test_pred_lr_lemma))


##### ---------------------------------------------------- Experiment 5: With Feature Selection (SelectKBest) ---------------------------------------#

# Select top K best features using chi-squared
k = 500  # You can adjust K to any number, such as 300, 500, or 1000
selector = SelectKBest(chi2, k=k)
X_train_vec_lemma_selected = selector.fit_transform(X_train_vec_lemma, y_train) # X_train_vec_lemma, X_test_vec_lemma similar to experiment 4
X_test_vec_lemma_selected = selector.transform(X_test_vec_lemma)

# Logistic Regression with Feature Selection
grid_search_logreg_lemma_selected = GridSearchCV(log_reg, param_grid_logreg, cv=strat_kfold)
grid_search_logreg_lemma_selected.fit(X_train_vec_lemma_selected, y_train)

# Best model with Feature Selection
best_log_reg_lemma_selected = grid_search_logreg_lemma_selected.best_estimator_
print("\nBest Logistic Regression Model (With Feature Selection):", best_log_reg_lemma_selected)

# Evaluation on test set with Feature Selection
y_test_pred_lr_lemma_selected = best_log_reg_lemma_selected.predict(X_test_vec_lemma_selected)
print("\nLogistic Regression (With Feature Selection) Test Accuracy:", accuracy_score(y_test, y_test_pred_lr_lemma_selected))
print(classification_report(y_test, y_test_pred_lr_lemma_selected))


##### --------------------------------------------------- Experiment 6 - Manual Hyperparameter tuning with Validation set -------------------------------#

# Combine and label the dataset
facts = [(fact, 1) for fact in real_facts] + [(fact, 0) for fact in fake_facts]
df = pd.DataFrame(facts, columns=['text', 'label'])

# Split the dataset into train (60%), validation (20%), and test (20%)
X_temp, X_test, y_temp, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

# Apply the enhanced preprocessing function from experiment 4
X_train_lemma = X_train.apply(preprocess_lemmatization_additional_steps)
X_val_lemma = X_val.apply(preprocess_lemmatization_additional_steps)
X_test_lemma = X_test.apply(preprocess_lemmatization_additional_steps)

# Feature extraction: TF-IDF for lemmatization data
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Using unigrams and bigrams
X_train_vec_lemma = vectorizer.fit_transform(X_train_lemma)
X_val_vec_lemma = vectorizer.transform(X_val_lemma)
X_test_vec_lemma = vectorizer.transform(X_test_lemma)

# Hyperparameter grid for Logistic Regression
param_grid_logreg = {'C': [0.1, 1, 10]}

# Function for manual tuning using validation set
best_val_score = 0
best_log_reg_model = None

for C in param_grid_logreg['C']:
    # Train Logistic Regression with the current C value
    log_reg = LogisticRegression(C=C)
    log_reg.fit(X_train_vec_lemma, y_train)

    # Evaluate on the validation set
    val_score = accuracy_score(y_val, log_reg.predict(X_val_vec_lemma))
    print(f"Logistic Regression with C={C}, Validation Accuracy: {val_score}")

    # Keep track of the best model
    if val_score > best_val_score:
        best_val_score = val_score
        best_log_reg_model = log_reg

# Best hyperparameters based on validation set
print("\nBest Logistic Regression Model based on Validation Set:", best_log_reg_model)

# Evaluation on the test set with the best model
y_test_pred_lr_lemma = best_log_reg_model.predict(X_test_vec_lemma)
print("\nLogistic Regression (Lemmatization + Extra Preprocessing) Test Accuracy:", accuracy_score(y_test, y_test_pred_lr_lemma))
print(classification_report(y_test, y_test_pred_lr_lemma))


#### ------------------------------------------------- Experiment 7: Manual Hyperparameter Tuning with cross validation --------------------------------------#

# Combine and label the dataset
facts = [(fact, 1) for fact in real_facts] + [(fact, 0) for fact in fake_facts]
df = pd.DataFrame(facts, columns=['text', 'label'])

# Split the dataset into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# Apply the enhanced preprocessing function from experiment 4
X_train_lemma = X_train.apply(preprocess_lemmatization_additional_steps)
X_test_lemma = X_test.apply(preprocess_lemmatization_additional_steps)

# Feature extraction: TF-IDF for lemmatization data
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Using unigrams and bigrams
X_train_vec_lemma = vectorizer.fit_transform(X_train_lemma)
X_test_vec_lemma = vectorizer.transform(X_test_lemma)

# Hyperparameter grid for Logistic Regression
param_grid_logreg = {'C': [0.1, 1, 10]}

# Stratified K-Fold Cross-Validation (using 5 folds)
strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_val_score = 0
best_log_reg_model = None

# Perform manual tuning with cross-validation
for C in param_grid_logreg['C']:
    log_reg = LogisticRegression(C=C)
    fold_scores = []

    for train_index, val_index in strat_kfold.split(X_train_vec_lemma, y_train):
        # Split training data into train/validation splits
        X_train_fold, X_val_fold = X_train_vec_lemma[train_index], X_train_vec_lemma[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # Train on the training folds
        log_reg.fit(X_train_fold, y_train_fold)

        # Validate on the validation fold
        val_score = accuracy_score(y_val_fold, log_reg.predict(X_val_fold))
        fold_scores.append(val_score)

    # Compute the average cross-validation score for this value of C
    avg_val_score = np.mean(fold_scores)
    print(f"Logistic Regression with C={C}, Cross-Validation Accuracy: {avg_val_score}")

    # Keep track of the best model
    if avg_val_score > best_val_score:
        best_val_score = avg_val_score
        best_log_reg_model = log_reg

# Best hyperparameters based on cross-validation
print("\nBest Logistic Regression Model based on Cross-Validation:", best_log_reg_model)

# Evaluation on the test set with the best model
y_test_pred_lr_lemma = best_log_reg_model.predict(X_test_vec_lemma)
print("\nLogistic Regression (Lemmatization + Extra Preprocessing) Test Accuracy:", accuracy_score(y_test, y_test_pred_lr_lemma))
print(classification_report(y_test, y_test_pred_lr_lemma))


