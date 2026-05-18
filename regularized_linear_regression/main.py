import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from ridge_lasso_model import RidgeRegression, LassoRegression


X, y = make_regression(
    n_samples=1000,
    n_features=50,
    noise=15,
    random_state=42
)

print("Dataset Shape:")
print("X Shape:", X.shape)
print("y Shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


ridge = RidgeRegression(
    learning_rate=0.001,
    epochs=2000,
    lambda_param=1
)

ridge.fit(X_train, y_train)

ridge_predictions = ridge.predict(X_test)

ridge_mse = mean_squared_error(y_test, ridge_predictions)
ridge_r2 = r2_score(y_test, ridge_predictions)

print("\nRIDGE REGRESSION RESULTS")
print("MSE:", ridge_mse)
print("R2 Score:", ridge_r2)


lasso = LassoRegression(
    learning_rate=0.001,
    epochs=2000,
    lambda_param=1
)

lasso.fit(X_train, y_train)

lasso_predictions = lasso.predict(X_test)

lasso_mse = mean_squared_error(y_test, lasso_predictions)
lasso_r2 = r2_score(y_test, lasso_predictions)

print("\nLASSO REGRESSION RESULTS")
print("MSE:", lasso_mse)
print("R2 Score:", lasso_r2)


plt.figure(figsize=(12, 5))
plt.plot(ridge.weights)
plt.title("Ridge Regression Coefficients")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.grid(True)
plt.savefig("outputs/ridge_coefficients.png")
plt.close()


plt.figure(figsize=(12, 5))
plt.plot(lasso.weights)
plt.title("Lasso Regression Coefficients")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.grid(True)
plt.savefig("outputs/lasso_coefficients.png")
plt.close()


models = ["Ridge", "Lasso"]
mse_values = [ridge_mse, lasso_mse]

plt.figure(figsize=(6, 5))
plt.bar(models, mse_values)
plt.title("MSE Comparison")
plt.ylabel("Mean Squared Error")
plt.savefig("outputs/mse_comparison.png")
plt.close()


ridge_non_zero = np.sum(ridge.weights != 0)
lasso_non_zero = np.sum(lasso.weights != 0)

print("\nNON-ZERO COEFFICIENTS")
print("Ridge:", ridge_non_zero)
print("Lasso:", lasso_non_zero)


print("\nPROJECT COMPLETED SUCCESSFULLY")