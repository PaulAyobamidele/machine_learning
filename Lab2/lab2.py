import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# load the wine datasets

wine = load_wine()


X = wine.data
y = wine.target


# convert data into a dataframe
df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = y


# perform ANOVA for feature selection
selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X, y)


# Get the indices of the selected features
selected_feature_indices = selector.get_support(indices=True)


# Get the names of the selected features
selected_features = df.columns[selected_feature_indices]


# print selected features
print("Selected Features:", selected_features)

print(df.head())
print(df['od280/od315_of_diluted_wines'])


# Toy datasets:
# Real-world datasets
# Generated datasets


# load_iris, load_digits, load_boston, load_diabetes, load_linnerud
# fetch_20newsgroups, fetch_openml, fetch_covtype, fetch_california_housing, and fetch_rcv1.
# make_classification, make_regression, make_blobs, make_moons, and make_circles.
