import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest


iris = load_iris()


X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df["target"] = y

print(df.head())

selector = SelectKBest(f_classif, k=3)
X_selected = selector.fit_transform(X, y)


selected_features_indices = selector.get_support(indices=True)

selected_feature = df.columns[selected_features_indices]

print("The selected features are:", selected_feature)


# CORRELATION
pearson_corr = df.corr(method="pearson")
print(pearson_corr)


most_correlated_features = (
    pearson_corr["target"].sort_values(ascending=False)[1:4].index
)

print("Features selected by SelectKBest:", selected_feature)
print("Most correlated features with the target:", most_correlated_features)


help(f_classif)
