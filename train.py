import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split
import extract_features as ef
import prepare_data as prep

print('Content of data/')
print(sorted(os.listdir('data/')))

label_mapping = "label:WillettsSpecific2018"

# Load and concatenate data from all files
data_list = []
for file in sorted(os.listdir('data/')):
    if not file.endswith('csv.gz'):
        continue
    data = pd.read_csv(f'data/{file}', index_col='time', parse_dates=['time'],
                       dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': 'string'})
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


# Split the data into training and evaluation sets (90% train, 10% eval)
X_train, X_eval, Y_train, Y_eval = train_test_split(X_feats, Y, test_size=0.1, random_state=0)

# Preprocess the features
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_eval_scaled = scaler.transform(X_eval)

# Define the model
clf = BalancedRandomForestClassifier(random_state=0)

# Cross-validation
cv_scores = cross_val_score(clf, X_train_scaled, Y_train, cv=5, scoring='accuracy')


# Print the results
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {cv_scores.mean()}')
print(f'Standard deviation of cross-validation score: {cv_scores.std()}')

# Fit the model on the training dataset
clf.fit(X_train_scaled, Y_train)

# Evaluate on the last dataset
X_eval_scaled = scaler.transform(X_eval)

# Predict and evaluate on the evaluation dataset
Y_pred = clf.predict(X_eval_scaled)
accuracy = np.mean(Y_pred == Y_eval)

print(f'Evaluation accuracy on the evaluation dataset: {accuracy}')

disp = ConfusionMatrixDisplay.from_predictions(
    Y_eval,
    Y_pred,
    display_labels=clf.classes_
)
disp.ax_.set_title("Confusion matrix")

print("Confusion matrix")
print(disp.confusion_matrix)

plt.show()