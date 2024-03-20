import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns

iris = datasets.load_iris()

iris.data.shape

iris.feature_names

df = pd.DataFrame(
    data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
)


df.head()

plt.figure(figsize=(12, 8))
sns.pairplot(df, hue="target", palette="viridis")
plt.suptitle("Scatterplot of Iris Dataset", y=1.02)
plt.show()


# Modelling

# Using one feature at a time
X = df["sepal length (cm)"].values.reshape(-1, 1)
y = df["target"].values


help(train_test_split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)

model.coef_
model.intercept_

y_pred = model.predict(X_test)
y_pred

plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color="black", label="Actual")
plt.plot(X_test, y_pred, color="coral", linewidth=2, label="Predicted")
plt.xlabel("Petak Width (cm)")
plt.ylabel("Species")
plt.title("Linear Regression on Iris Dataset")
plt.legend()
plt.show()


from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Convert predcted values to integer (rounding to nearest)
y_pred_int = np.round(y_pred).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_int)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_int)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


# Using Second Feature

X = df["sepal width (cm)"].values.reshape(-1, 1)
y = df["target"].values


# Separate the train and test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

y_pred


# Count predicted values to integer (rounding to nearest)

y_pred_int = np.round(y_pred).astype(int)

# check accuracy
accuracy = accuracy_score(y_test, y_pred_int)


# calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_int)


# plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


# Using third Feature

X = df["petal length (cm)"].values.reshape(-1, 1)
y = df["target"].values


# Seperate train and test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# Predict unknown data

y_pred

# Convert predicted values to integer (rounding to nearest)
y_pred_int = np.round(y_pred).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_int)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_int)

# plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


# Using Fourth Feature

X = df["petal width (cm)"].values.reshape(-1, 1)
y = df["target"].values


# seperate the train and test sets
from sklearn.model_selection import train_test_split

# split the data

help(train_test_split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred

y_pred_int = np.round(y_pred).astype(int)

accuracy = accuracy_score(y_test, y_pred_int)

cm = confusion_matrix(y_test, y_pred_int)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
df = pd.DataFrame(
    data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
)

# select features and target
X = df.drop("target", axis=1)
y = df["target"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

y_pred_int = np.round(y_pred).astype(int)

f1 = f1_score(y_test, y_pred_int, average="weighted")
print("F1 score with all features: ", f1)

cm = confusion_matrix(y_test, y_pred_int)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


# Bruteforce Search

from itertools import combinations
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def brute_force_feature_selection(X, y, model, tolerance=0.95):
    """
    Perform brute-force feature selection by iterating over all possible combinations of features.

    Args:
        X (_type_): _description_
        y (_type_): _description_
        model (_type_): _description_
        tolerance (float, optional): _description_. Defaults to 0.95.
    """

    num_features = X.shape[1]
    feature_names = [f"Feature {i+1}" for i in range(num_features)]
    results = []

    # Iterate over all possible combinations of features
    for r in range(1, num_features + 1):
        for combination in combinations(range(num_features), r):
            X_subset = X[:, combination]
            X_train, X_test, y_train, y_test = train_test_split(
                X_subset, y, test_size=0.2, random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, np.round(y_pred).astype(int))

            results.append([list(combination), accuracy])

        results_df = pd.DataFrame(results, columns=["Used Features", "Accuracy"])
        return results_df


iris = load_iris()
X = iris.data
y = iris.target
model = LinearRegression()
results_df = brute_force_feature_selection(X, y, model, tolerance=0.95)
sorted_results_df = results_df.sort_values(by="Accuracy", ascending=False)
sorted_results_df


# Using Recursive Feature Elimination

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the Linear Regression model
model = LinearRegression()


# Recursive Feature Elimination (RFE) from scratch with accuracy
def custom_rfe(model, X_train, X_test, y_train, y_test):
    num_features = X_train.shape[1]
    selected_features = list(range(num_features))
    best_accuracy = 0
    best_feature_set = selected_features.copy()
    while len(selected_features) > 1:
        worst_feature = None
        min_coefficient = float("inf")
        for feature in selected_features:
            features_subset = selected_features.copy()
            features_subset.remove(feature)
            X_train_subset = X_train[:, features_subset]
            X_test_subset = X_test[:, features_subset]
            model.fit(X_train_subset, y_train)
            y_pred = model.predict(X_test_subset)
            accuracy = accuracy_score(y_test, np.round(y_pred).astype(int))
            print("features_subset : ", features_subset, "accuracy: ", accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature_set = features_subset.copy()
            coefficients = np.abs(model.coef_)
            if np.min(coefficients) < min_coefficient:
                min_coefficient = np.min(coefficients)
                worst_feature = feature
        print("worst_feature: ", worst_feature)
        selected_features.remove(worst_feature)
    return best_feature_set


# Perform RFE to select features
selected_features = custom_rfe(model, X_train, X_test, y_train, y_test)
print("Selected Features:", selected_features)

# Transform the training and testing datasets to include only the selected features
X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]

# Train the Linear Regression model with the selected features
model.fit(X_train_selected, y_train)

# Make predictions
y_pred = model.predict(X_test_selected)

# Calculate Accuracy
accuracy = accuracy_score(y_test, np.round(y_pred).astype(int))
print("Accuracy:", accuracy)


# Using the python predefined function

from sklearn.feature_selection import RFE

help(RFE)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.datasets import load_iris


# load iris datasets

iris = load_iris()

df = pd.DataFrame(
    data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
)

# Select features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

# perform recursive feature elimination
rfe = RFE(estimator=model, n_features_to_select=3, step=1)
rfe.fit(X_train, y_train)


# get the features
selected_features = X.columns[rfe.support_]
print("The features are:", selected_features)

model.fit(X_train[selected_features], y_train)

y_pred = model.predict(X_test[selected_features])

y_pred_int = np.round(y_pred).astype(int)

f1 = f1_score(y_test, y_pred_int, average="weighted")
print("F1 score with selected features: ", f1)


# Calculate the confusion matrix

cm = confusion_matrix(y_test, y_pred_int)

# Plot confusion matix
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


# RFE on diabetes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.datasets import load_diabetes


db = load_diabetes()

df = pd.DataFrame(
    data=np.c_[db["data"], db["target"]], columns=db["feature_names"] + ["target"]
)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)


rfe = RFE(estimator=model, n_features_to_select=3, step=1)

rfe.fit(X_train, y_train)


selected_features = X.columns[rfe.support_]

model.fit(X_train[selected_features], y_train)

y_pred = model.predict(X_test[selected_features])


y_pred_int = np.round(y_pred).astype(int)

# f1 = f1_score(y_test, y_pred_int, average="weighted")
mse = mean_squared_error(y_test, y_pred_int)

print(mse)

# cm = confusion_matrix(y_test, y_pred_int)


plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=np.unique(y_test),
    yticklabels=np.unique(y_test),
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
