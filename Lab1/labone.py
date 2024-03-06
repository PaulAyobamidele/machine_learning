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


sns.heatmap(pearson, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()


# Load the DataFrame
h = pd.read_csv('Lab1/house_price_dataset_train_kaggle.csv')


print(h)
print(h.dtypes)
print(len(h.columns))

h = h.dropna(how='all')
print(h)

h_dropped = h.drop(h.select_dtypes(['object']), axis=1)
print(h_dropped)

correlation = h.corr('pearson')
print(correlation)

for col in correlation.columns:
    print(col, pd.api.types.is_numeric_dtype(correlation[col]))
