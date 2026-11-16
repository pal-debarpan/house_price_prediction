from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns

housing = fetch_california_housing(as_frame=True)
df = housing.frame

plt.figure(figsize=(10, 5))

sns.histplot(data=df, x='MedHouseVal', bins=50, kde=True)

plt.title("Distribution of California House Values")
plt.xlabel("Median House Value ($100,000s)")
plt.ylabel("Number of Districts")

plt.show()