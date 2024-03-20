"""Standardize the d-dimensional dataset: This step involves making sure that all the features (dimensions) in your dataset have the same scale. We do this to prevent certain features from dominating others simply because they have larger values. Standardizing means transforming the data such that each feature has a mean of 0 and a standard deviation of 1.

Construct the covariance matrix: The covariance matrix tells us how different features vary with respect to each other. It's like a measure of how much two variables change together. In PCA, we use the covariance matrix to understand the relationships between different features in our dataset.

Decompose the covariance matrix into its eigenvectors and eigenvalues: This is where things get a bit mathematical. Eigenvectors are special directions in the feature space, and eigenvalues represent the magnitude of variance in those directions. Essentially, eigenvectors are the directions of maximum variance in the data, and eigenvalues represent the amount of variance in those directions.

Sort the eigenvalues by decreasing order to rank the corresponding eigenvectors: We arrange the eigenvalues in descending order. This step helps us prioritize which directions (or eigenvectors) contain the most information about the data.

Select k eigenvectors which correspond to the k largest eigenvalues: Here, we decide how many principal components (new features) we want to keep. Typically, we select the top k eigenvectors that correspond to the largest eigenvalues. These eigenvectors represent the most important directions in the data.

Construct a projection matrix W from the "top" k eigenvectors: The projection matrix is formed by stacking the selected eigenvectors as columns. This matrix defines the new coordinate system in which we'll represent our data.

Transform the d-dimensional input dataset X using the projection matrix W to obtain the new k-dimensional feature subspace: Finally, we multiply our original dataset by the projection matrix. This effectively rotates and projects the data onto the new feature subspace defined by the selected principal components.

Overall, PCA helps us reduce the dimensionality of our dataset while preserving as much information as possible. It achieves this by finding the most important directions (principal components) along which the data varies the most.
"""

"""
Decompose the covariance matrix into its eigenvectors and eigenvalues:

Imagine you have a dataset with multiple features, and you want to understand how these features are related to each other. The covariance matrix helps us quantify these relationships.

Here's what happens in this step:

Covariance Matrix: The covariance matrix is a square matrix where each element represents the covariance between two features. For example, if we have features X and Y, the element at the ith row and jth column of the covariance matrix represents the covariance between feature i and feature j.

Eigenvectors and Eigenvalues: When we decompose the covariance matrix, we're essentially breaking it down into its fundamental building blocks. The decomposition reveals two important things: eigenvectors and eigenvalues.

Eigenvectors: These are special vectors that don't change their direction when a linear transformation is applied to them, but they might only be scaled. In the context of PCA, these eigenvectors represent the principal components, or the directions of maximum variance in the data.

Eigenvalues: These are scalar values that indicate the magnitude of the variance in the corresponding eigenvectors' directions. Higher eigenvalues mean more variance along the corresponding eigenvector, making it a more important direction in the data.

What it Means: Decomposing the covariance matrix into eigenvectors and eigenvalues essentially helps us find the most meaningful directions (eigenvectors) in our dataset along with their importance (eigenvalues). These directions capture the most variation present in the data.

Why it's Important: By identifying these principal components (eigenvectors) and their associated variance (eigenvalues), we can prioritize which features contribute the most to the variability in the dataset. This knowledge is crucial for dimensionality reduction because it allows us to retain the most important features while discarding those with less impact.

In essence, step 3 of PCA helps us understand the underlying structure of our data by revealing the directions along which the data varies the most (eigenvectors) and how much variability there is in those directions (eigenvalues).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
    header=None,
)

df.head()


# standardize the features
X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
scaler = StandardScaler()
X_train_stand = scaler.fit_transform(X_train)
X_test_stand = scaler.transform(X_test)


# obtaining the eigen vector and eigen values

cov_mat = np.cov(X_train_stand.T)

eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print("\nEigenvalues \n%s" % eigen_vals)


tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.bar(
    range(1, 14),
    var_exp,
    alpha=0.5,
    align="center",
    label="individual explained variance",
)
plt.step(range(1, 14), cum_var_exp, where="mid", label="cumulative explained variance")
plt.ylabel("Explained variance ratio")
plt.xlabel("Principal component index")
plt.legend(loc="best")
plt.show()


eigen_pairs = [
    (np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))
]


eigen_pairs.sort(key=lambda k: k[0], reverse=True)


w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
print("Matrix W:\n", w)


X_train_stand[0].dot(w)


X_train_pca = X_train_stand.dot(w)


colors = ["r", "b", "g"]
markers = ["s", "x", "o"]
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(
        X_train_pca[y_train == l, 0],
        X_train_pca[y_train == l, 1],
        c=c,
        label=l,
        marker=m,
    )
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend(loc="lower left")
plt.show()
