import pandas as pd
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


boston = load_boston()

for key, value in boston.items():
    print("Key: {}, Value: {}, Data Type: {}".format(key, value, type(value)))

X = boston.data
y = boston.target


df = pd.DataFrame(X, columns=boston.feature_names)
df['target'] = y


selector = SelectKBest(f_classif, k=3)
X_selected = selector.fit_transform(X, y)


selected_features_indices = selector.get_support(indices=True)

selected_features = df.columns[selected_features_indices]

print(selected_features)

for i in selected_features:
    print("{}: datatypes is {}".format(i, type(i)))


# CORRELATION

correlation = df.corr(method='pearson')


print(correlation)

most_correlated = correlation['target'].sort_values(ascending=False)[1:4].index

print("Features Based on Pearson Correlation:", most_correlated)
print("Features Based on Anova", selected_features)


# Different methodologies: ANOVA (Analysis of Variance) and correlation coefficient are different statistical techniques used for different purposes. ANOVA is typically used to compare means across multiple groups, whereas correlation measures the strength and direction of a linear relationship between two variables.

# Different assumptions: ANOVA assumes that the data is normally distributed and that the groups being compared have equal variances. On the other hand, Pearson correlation assumes that the data is linearly related and follows a bivariate normal distribution.

# Sensitivity to data distribution: ANOVA might select features that show significant differences in means across groups, even if they are not strongly correlated with the target variable. Pearson correlation, on the other hand, specifically measures the linear relationship between variables.

# Multicollinearity: Pearson correlation detects linear relationships between pairs of variables, but it doesn't consider multicollinearity, which occurs when two or more predictors in a regression model are highly correlated. ANOVA might select features that are less correlated but still provide significant information for predicting the target variable.

# Target variable relationship: ANOVA selects features that show significant differences in means across groups defined by the target variable. Pearson correlation measures the strength of the linear relationship between each predictor and the target variable individually.
