import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import seaborn as sns


iris = load_iris()
iris_pandas = pd.DataFrame(data=iris.data, columns=iris.feature_names)


iris_pandas.head()
iris_pandas.info()
iris_pandas.tail()
iris_pandas.describe()

pearson = iris_pandas.corr()
print(pearson)


sns.heatmap(pearson, annot=True, cmap="coolwarm", fmt=".2f")
plt.show()


h_pd = pd.read_csv("house_price_dataset_train_kaggle.csv")
h_pd


# for column in h_pd.columns:
#     if h_pd.api.types.is_numeric_dtype(h_pd[column]):
#         h_pd = h_pd[pd.to_numeric(h_pd[column], errors='coerce').notnull()]

# h_pd = h_pd.reset_index(drop=True)

# cleaned_h_pd = h_pd.dropna(axis=1, how='all')

print(cleaned_h_pd)


correlation = h_pd.corr()
print(correlation)


import pandas as pd

# Create a DataFrame
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})

# Selecting the first row and second column
print(df.iloc[0, 1])  # Output: 4

# Selecting rows 0 to 1 and all columns
print(df.iloc[0:2, :])  # Output:
#    A  B  C
# 0  1  4  7
# 1  2  5  8

# Selecting all rows and the second column
print(df.iloc[:, 1])  # Output:
# 0    4
# 1    5
# 2    6
# Name: B, dtype: int64
