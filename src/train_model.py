from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prepare_data import X_train, y_train, X_test, y_test
import matplotlib.pyplot as plt
import numpy as np

model = LinearRegression()

model.fit(X_train, y_train)

print("Model training complete.")

predictions = model.predict(X_test)

'''
plt.scatter(y_test, predictions)
plt.xlabel("Actual House Values")
plt.ylabel("Predicted House Values")
plt.title("Actual vs Predicted House Values")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linestyle="--"
)
plt.show()
'''

#mae = mean_absolute_error(predictions, y_test)
#print(mae)

#mse = mean_squared_error(y_test, predictions)
#print(mse)

#r2 = r2_score(y_test, predictions)
#print(r2)

residuals = y_test - predictions

plt.scatter(predictions, residuals)
plt.xlabel("Predictions")
plt.ylabel("Residuals")
plt.axhline(y=0, color="red", linestyle="--")
plt.title("Predicitons vs Residuals")
plt.show()

print(np.mean(residuals))