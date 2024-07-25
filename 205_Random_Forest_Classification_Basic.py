## import libraries
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


my_df = pd.read_csv("data/sample_data_classification.csv")

# Split data into input and output variables
X = my_df.drop(["output"], axis=1)
y = my_df["output"]

## Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    stratify=y, random_state=42)

# Instantiate model
clf = RandomForestClassifier(random_state=42)

# Train model
clf.fit(X_train, y_train)

# Acsess accuracy
y_pred = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)

accuracy_score(y_test, y_pred)

precision_score(y_test, y_pred)

recall_score(y_test, y_pred) 