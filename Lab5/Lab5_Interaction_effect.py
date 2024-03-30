"""Interaction Effects
1.Boston housing dataset, a popular choice for regression problems. This exercise will guide you through creating an interaction term between two predictors and analyzing its impact on a linear regression model predicting housing prices.

2.  Goal: Investigate how the interaction between average number of rooms per dwelling (RM) and proportion of owner-occupied units built before 1940 (AGE) affects the median value of owner-occupied homes (MEDV).

3.  Dataset: The Boston housing dataset contains information collected by the U.S Census Service concerning housing in the area of Boston Mass. It has 506 entries with 13 features and one target variable (MEDV: Median value of owner-occupied homes in $1000).
"""

# Load the Dataset: Use scikit-learn to load the Boston housing agent
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
boston = pd.read_csv("boston_house_prices.csv", skiprows=1)

# Split the data into features (X) and target variable (y)
X = boston.drop(columns="MEDV")
y = boston["MEDV"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Calculate the interaction term RM_AGE_interaction using the original data
boston_interaction = boston.copy()
boston_interaction["RM_AGE_interaction"] = (
    boston_interaction["RM"] * boston_interaction["AGE"]
)

# Split the data into features and target variable
X_interaction = boston_interaction.drop(columns="MEDV")
y_interaction = boston_interaction["MEDV"]

# Split the data into training and testing sets
X_train_interaction, X_test_interaction, y_train_interaction, y_test_interaction = (
    train_test_split(X_interaction, y_interaction, test_size=0.2, random_state=42)
)

# Instantiate and train the linear regression model without the interaction term
model_without_interaction = LinearRegression()
model_without_interaction.fit(X_train, y_train)

# Instantiate and train the linear regression model with the interaction term
model_with_interaction = LinearRegression()
model_with_interaction.fit(X_train_interaction, y_train_interaction)

# Make predictions on the testing data for both models
y_pred_without_interaction = model_without_interaction.predict(X_test)
y_pred_with_interaction = model_with_interaction.predict(X_test_interaction)

# Evaluate the performance of both models using R^2 score
r2_without_interaction = round(r2_score(y_test, y_pred_without_interaction), 4)
r2_with_interaction = round(r2_score(y_test_interaction, y_pred_with_interaction), 4)

print("R^2 Score without Interaction Term:", r2_without_interaction)
print("R^2 Score with Interaction Term:", r2_with_interaction)


# Correlation

boston = boston.corr("pearson")


most_correlated_features = boston["MEDV"].sort_values(ascending=False)[1:5].index

features = ["RM", "LSTAT", "TAX", "INDUS"]
target = "MEDV"


for feature in features:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=boston, x=feature, y=target, hue="AGE", palette="coolwarm")
    plt.title(f"{feature} vs. {target} by AGE")

# boston = load_boston()
# df = pd.DataFrame(boston.data, columns=boston.feature_names)
# df['MEDV'] = boston.target


plt.figure(figsize=(14, 10))
sns.heatmap(boston.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.show()


# LSAT, RM, INDUS, TAX
