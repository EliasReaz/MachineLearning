## import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# import data
my_df = pd.read_csv("data/sample_data_classification.csv")

# split data into input and output variables
X = my_df.drop("output", axis=1)
y = my_df["output"]

## split train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate model
clf = LogisticRegression()

# train model
clf.fit(X_train, y_train)

# access model accuracy
y_pred = clf.predict(X_test)
print(f"y_pred: {y_pred}")

y_pred_prob = clf.predict_proba(X_test)
print(f"y_pred_prob: {y_pred_prob}")

accuracy_score(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# conf_matix_df = pd.DataFrame(conf_matrix)
# print(conf_matix_df)

# plt.style.use("seaborn-poster")
plt.matshow(conf_matrix, cmap="coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
for (i,j), corr_val in np.ndenumerate(conf_matrix):
    plt.text(j,i, corr_val, ha="center", va="center", fontsize=20)
plt.show()