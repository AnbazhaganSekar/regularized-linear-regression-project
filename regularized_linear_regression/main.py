import os
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
    epochs=3000,
    lambda_param=10
)

lasso.fit(X_train, y_train)

lasso_predictions = lasso.predict(X_test)

lasso_mse = mean_squared_error(y_test, lasso_predictions)
lasso_r2 = r2_score(y_test, lasso_predictions)

print("\nLASSO REGRESSION RESULTS")
print("MSE:", lasso_mse)
print("R2 Score:", lasso_r2)

print("\nRIDGE COEFFICIENTS")
print(ridge.weights)

print("\nLASSO COEFFICIENTS")
print(lasso.weights)

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(12, 5))
plt.plot(ridge.weights)
plt.title("Ridge Regression Coefficients")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ridge_coefficients.png"))
plt.close()


plt.figure(figsize=(12, 5))
plt.plot(lasso.weights)
plt.title("Lasso Regression Coefficients")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "lasso_coefficients.png"))
plt.close()


models = ["Ridge", "Lasso"]
mse_values = [ridge_mse, lasso_mse]

plt.figure(figsize=(6, 5))
plt.bar(models, mse_values)
plt.title("MSE Comparison")
plt.ylabel("Mean Squared Error")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mse_comparison.png"))
plt.close()


ridge_non_zero = np.sum(ridge.weights != 0)
lasso_non_zero = np.sum(np.abs(lasso.weights) > 0.01)

print("\nNON-ZERO COEFFICIENTS")
print("Ridge:", ridge_non_zero)
print("Lasso:", lasso_non_zero)


lambda_values = [0.01, 0.1, 1, 10, 100]

non_zero_coefficients = []

for lam in lambda_values:

    temp_lasso = LassoRegression(
        learning_rate=0.001,
        epochs=3000,
        lambda_param=lam
    )

    temp_lasso.fit(X_train, y_train)

    count = np.sum(np.abs(temp_lasso.weights) > 0.01)

    non_zero_coefficients.append(count)


plt.figure(figsize=(8,5))

plt.plot(
    lambda_values,
    non_zero_coefficients,
    marker='o'
)

plt.xscale("log")

plt.title("Lambda vs Non-Zero Coefficients")

plt.xlabel("Lambda")

plt.ylabel("Non-Zero Coefficients")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "lambda_sparsity_analysis.png")
)

plt.close()

print("\nPROJECT COMPLETED SUCCESSFULLY")