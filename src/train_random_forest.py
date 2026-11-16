from sklearn.ensemble import RandomForestRegressor
from prepare_data import X_train,X_test,y_train,y_test
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


model = RandomForestRegressor(n_estimators=500, random_state=42)

model.fit(X_train, y_train)

print("Random Forest Training Completed")

y_pred = model.predict(X_test)

print('test:', r2_score(y_test, y_pred))

train_pred = model.predict(X_train)
print('train:', r2_score(y_train, train_pred))

mae = mean_absolute_error(y_test, y_pred)
print('MAE:', mae)

mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)

rmse = np.sqrt(mse)
print('RMSE:', rmse)

residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='red')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted House Prices")

plt.plot([0, 5], [0, 5], color='red')
plt.show()