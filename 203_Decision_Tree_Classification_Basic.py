
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


my_df = pd.read_csv("data/sample_data_classification.csv")

# Split data into input and output variables
X = my_df.drop(["output"], axis=1)
y = my_df["output"]

## Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    stratify=y, random_state=42)

# Instantiate model
clf = DecisionTreeClassifier(random_state=42, min_samples_leaf=7)

# Train model
clf.fit(X_train, y_train)

# Acsess accuracy
y_pred = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)

accuracy_score(y_test, y_pred)

precision_score(y_test, y_pred)

recall_score(y_test, y_pred) 

conf_matrix = confusion_matrix(y_test, y_pred) 
print(conf_matrix)
# plot tree
plt.figure(figsize=(25,15))
tree = plot_tree(clf, feature_names=X.columns,
                 filled=True,
                 rounded=True,
                 fontsize=24)

plt.matshow(conf_matrix, cmap="coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
for (i,j), val in np.ndenumerate(conf_matrix):
    # print((i,j), val)
    plt.text(j,i, val, ha="center", va="center", fontsize=20)
plt.tight_layout()
plt.show()
