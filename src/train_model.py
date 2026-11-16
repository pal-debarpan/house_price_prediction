from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prepare_data import X_train, y_train, X_test, y_test

model = LinearRegression()

model.fit(X_train, y_train)

print("Model training complete.")

predictions = model.predict(X_test)

mae = mean_absolute_error(predictions, y_test)
print(mae)

mse = mean_squared_error(y_test, predictions)
print(mse)

r2 = r2_score(y_test, predictions)
print(r2)