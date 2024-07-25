#########################################
# PCA - Code Stencil
##########################################

########################################################################
## import libraries
########################################################################
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
# Recursive Feature Elemination
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

########################################################################
## import data
########################################################################

# import data
data_for_modeling = pd.read_csv("data/sample_data_pca.csv")
print(data_for_modeling.columns)
print(data_for_modeling.head())

# Drop uncessary columns
data_for_modeling.drop(["user_id"], axis=1, inplace=True)

# shuffle data
data_for_modeling = shuffle(data_for_modeling, random_state=42)

#############################
## see class balance
#############################
data_for_modeling["purchased_album"].value_counts(normalize=True)

########################################################################
## Deal with missing values
########################################################################
print(data_for_modeling.isna().sum())
print(data_for_modeling.isna().sum().sum())
## only few missing values, so we drop rows with any missing value
data_for_modeling.dropna(how="any", inplace=True)

########################################################################
## Split input variables and output variables
########################################################################

X = data_for_modeling.drop("purchased_album", axis=1)
y = data_for_modeling["purchased_album"]

########################################################################
## Split out train and test set
########################################################################

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, 
                                                    random_state=42, stratify=y)

##############################################
# Feature Scaling
##############################################

scaled_standard = StandardScaler()
X_train_scaled = scaled_standard.fit_transform(X_train)
X_test_scaled = scaled_standard.transform(X_test)

################################################
# Apply PCA
#################################################

# instantiate & fit
pca = PCA(n_components=None, random_state=42)
pca.fit(X_train_scaled)


# Extract explained variance across components

explained_variance_ratio = pca.explained_variance_ratio_
explained_variance_ratio_cumulative = pca.explained_variance_ratio_.cumsum() 

###################################################
## plot explained variance across components
####################################################

plt.figure(figsize=(15,10))
plt.subplot(2,1,1)
plt.bar(list(range(1, len(explained_variance_ratio)+1)), explained_variance_ratio)
plt.xlabel("Number of Components")
plt.ylabel("% variance")
plt.title("Variance across principal components")
plt.tight_layout()

plt.subplot(2,1,2)
plt.plot(list(range(1, len(explained_variance_ratio)+1)), 
         explained_variance_ratio_cumulative)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative % variance")
plt.title("Cumulative variance across principal components")
plt.tight_layout()
plt.show()


#######################################################
## Apply PCA with selected number of components
#######################################################

pca = PCA(n_components=0.75, random_state=42)
X_train_1 = pca.fit_transform(X_train_scaled)
X_test_1 = pca.transform(X_test_scaled)

###################################################
# Apply RandomForestClassifier
####################################################

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_1, y_train)

####################################################
## Access model accuracy
#####################################################

y_pred = clf.predict(X_test_1)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {round(accuracy,4)}")


