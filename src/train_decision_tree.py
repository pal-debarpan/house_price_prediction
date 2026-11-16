from sklearn.tree import DecisionTreeRegressor
from prepare_data import X_train,X_test,y_test,y_train
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model = DecisionTreeRegressor(max_depth=20, random_state=42)

model.fit(X_train, y_train)

print('Model Training Done')

predictions = model.predict(X_test)

print(predictions[:10])

print(y_test[:10].values)

mae = mean_absolute_error(predictions, y_test)

mse = mean_squared_error(predictions, y_test)

r2 = r2_score(predictions, y_test)

train_predictions = model.predict(X_train)

r2_train = r2_score(y_train, train_predictions)

print('test r2:', r2)
print('train r2:', r2_train)