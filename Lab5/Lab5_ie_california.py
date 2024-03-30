import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the dataset
california = fetch_california_housing()

X = california.data
y = california.target


# Creat the Dataframe
df = pd.DataFrame(X, columns=california.feature_names)
df["target"] = y


# Find the best features

feature_names = [df.columns]

corr = df.corr("pearson")

top_features = corr["target"].sort_values(ascending=False)[1:3].index
# Split the Data

df["MedInc_AveRooms"] = df["MedInc"] * df["AveRooms"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Build your model

lr = LinearRegression()

lr.fit(X_train, y_train)

# Make your prediction
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

# Evaluate the Prediction

accuracy_r2 = r2_score(y_train, y_lr_train_pred)
accuracy_r2 = r2_score(y_test, y_lr_test_pred)


mse = mean_squared_error
accuracy_mse = mse(y_train, y_lr_train_pred)
accuracy_mse = mse(y_test, y_lr_test_pred)
