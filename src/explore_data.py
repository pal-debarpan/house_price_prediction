from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)

df = housing.frame

print(df.head())

print(df.shape)

print(df.columns)

df.info()

print(df.describe())