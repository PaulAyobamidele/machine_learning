import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


breast_cancer = load_breast_cancer()

print(breast_cancer.keys())

# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

X = breast_cancer.data
y = breast_cancer.target

df = pd.DataFrame(X, columns=breast_cancer.feature_names)
df["target"] = y

selector = SelectKBest(chi2, k=1)
X_selected = selector.fit_transform(X, y)


selected_indices = selector.get_support(indices=True)

selected_features = df.columns[selected_indices]


print(selected_indices)
print(selected_features)


# Correlation

breast_corr = df.corr("spearman")

print(breast_corr.columns)

most_correlated = breast_corr["target"].sort_values(ascending=False)[1:2].index

print(most_correlated)
print(selected_features)
