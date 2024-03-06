import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest


diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

df = pd.DataFrame(X, columns=diabetes.feature_names)
df['target'] = y

selector = SelectKBest(f_classif, k=2)
X_selected = selector.fit_transform(X, y)


selected_feature_indices = selector.get_support(indices=True)


selected_feature = df.columns[selected_feature_indices]

print("The features are:", selected_feature)


# CORRELATION
pearson = df.corr()

print(pearson)


most_correlated_features = pearson['target'].sort_values(ascending=False)[
    1:3].index

print("The features from the correlation:", most_correlated_features)
print("The features from the ANOVA:", selected_feature)
