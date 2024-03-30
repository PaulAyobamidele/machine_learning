import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

boston = pd.read_csv("boston_house_prices.csv", skiprows=1)


# Create the interaction feature LSTAT * RM]

boston["LSTAT_RM"] = boston["LSTAT"] * boston["RM"]

X = boston.drop(columns="MEDV")
y = boston["MEDV"]

# split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and fit the model
new_model_with_interaction = LinearRegression()
new_model_with_interaction.fit(X_train, y_train)


y_lr_train_pred = new_model_with_interaction.predict(X_train)
y_lr_test_pred = new_model_with_interaction.predict(X_test)


mse_accuracy_train = mean_squared_error(y_train, y_lr_train_pred)
mse_accuracy_test = mean_squared_error(y_test, y_lr_test_pred)


r2_score_train = r2_score(y_train, y_lr_train_pred)
r2_score_test = r2_score(y_test, y_lr_test_pred)


mse_accuracy = pd.DataFrame(
    [
        "Linear_Regression",
        mse_accuracy_train,
        mse_accuracy_test,
        r2_score_train,
        r2_score_test,
    ]
).transpose()

mse_accuracy.columns = [
    "Model",
    "mse_accuracy_train",
    "mse_accuracy_test",
    "r2_score_train",
    "r2_score_test",
]


mse_accuracy
