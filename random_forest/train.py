import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss, ConfusionMatrixDisplay
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split
from random_forest import extract_features as ef, prepare_data as prep
from tqdm import tqdm

# Check the contents of the data directory
print('Content of data/')
print(sorted(os.listdir('data/')))

# Define the label mapping
label_mapping = "label:Willetts2018"

# Load the metadata file
metadata = pd.read_csv('data/metadata.csv')

# Load and concatenate data from all files
data_list = []
for file in sorted(os.listdir('data/')):
    if not file.endswith('csv.gz'):
        continue

    data = pd.read_csv(f'data/{file}', index_col='time', parse_dates=['time'],
                       dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': 'string'})

    file_id = file.split('.')[0]
    print('Current file:', file_id)

    # Find the metadata for this file
    file_metadata = metadata[metadata['pid'] == file_id]

    # Add metadata to the data
    if not file_metadata.empty:
        for col in ['age', 'sex']:
            if col in file_metadata.columns:
                data[col] = file_metadata.iloc[0][col]

    # Map the label
    prep.map_labels(data, prep.anno_label_dict, label_mapping)
    data_list.append(data)

X = []
Y = []

for data in data_list:
    X_, Y_ = prep.extract_windows(data)
    X.append(X_)
    Y.append(Y_)

X_feats = []

for X_ in X:
    X_feats.append(pd.DataFrame([ef.extract_features(x) for x in X_]))

X_feats = pd.concat(X_feats, ignore_index=True)
Y = np.concatenate(Y, axis=0)

print('Data preprocessing done.\n')

# Split the data into training and evaluation sets (90% train, 10% eval)
X_train, X_eval, Y_train, Y_eval = train_test_split(X_feats, Y, test_size=0.1, random_state=0)

# Preprocess the features
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_eval_scaled = scaler.transform(X_eval)

# Define the model
clf = BalancedRandomForestClassifier(random_state=0, max_depth=None, max_features='sqrt', n_estimators=300,
                                     min_samples_leaf=1, min_samples_split=2, sampling_strategy='all', replacement=True, bootstrap=False)

# Cross-validation with progress bar
cv_scores = []
for _ in tqdm(range(5), desc="Cross-validation progress"):  # Progress bar for CV
    score = cross_val_score(clf, X_train_scaled, Y_train, cv=5, scoring='accuracy')
    cv_scores.append(score)

cv_scores = np.array(cv_scores)

# Print the results
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {cv_scores.mean()}')
print(f'Standard deviation of cross-validation score: {cv_scores.std()}')

# Fit the model on the training dataset
clf.fit(X_train_scaled, Y_train)

# Evaluate on the last dataset
X_eval_scaled = scaler.transform(X_eval)
Y_pred = clf.predict(X_eval_scaled)

# Calculate the important metrics
accuracy = accuracy_score(Y_eval, Y_pred)
f1_score = f1_score(Y_eval, Y_pred, average='weighted')
precision = precision_score(Y_eval, Y_pred, average='weighted')
recall = recall_score(Y_eval, Y_pred, average='weighted')
log_loss = log_loss(Y_eval, Y_pred)

print(f'Accuracy: {accuracy}')
print(f'F1 score: {clf.score(X_eval_scaled, Y_eval)}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Log loss: {log_loss}')
print(f'Confusion matrix: {clf.classes_}')

disp = ConfusionMatrixDisplay.from_predictions(
    Y_eval,
    Y_pred,
    display_labels=clf.classes_
)
disp.ax_.set_title("Confusion matrix")

print("Confusion matrix")
print(disp.confusion_matrix)

plt.show()
