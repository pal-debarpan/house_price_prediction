from sklearn.datasets import fetch_california_housing
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

housing = fetch_california_housing(as_frame=True)

df = housing.frame

correlation = df.corr()

print(correlation)

sns.heatmap(correlation, annot=True)

plt.title("HeatMap")
plt.show()