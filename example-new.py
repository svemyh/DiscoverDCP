import numpy as np
from pysr import PySRRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sympy

# Set random seed for reproducibility
np.random.seed(42)

def hidden_model(X):
    # Hidden convex function: f(x) = exp(x1 + x2) + (x3)^2 + 3*x4 + 2
    return np.exp(X[:, 0] + X[:, 1]) + np.square(X[:, 2]) + 3 * X[:, 3] + 2

n_samples = 500
n_features = 4
X = np.random.randn(n_samples, n_features)
y = hidden_model(X)

# Optional: Add some noise to the data
y += 0.1 * np.random.randn(n_samples)


def sympy_max(x, y):
    return sympy.Max(x, y)

def sympy_identity(x):
    return x

binary_operators = ["+", "max"]
unary_operators = ["exp", "square", "identity"]
extra_sympy_mappings = {
    "max": sympy_max,
    "identity": sympy_identity,
    "square": lambda x: x**2,
}
constraints = {
    "exp": 5,
    "square": 5,
    "identity": 1,
    "max": (-1, -1),
}
nested_constraints = {
    "exp": {"exp": 0, "square": 1, "max": 0},
    "square": {"exp": 0, "square": 0, "max": 0},
    "max": {"exp": 0, "square": 0, "max": 0},
}

model = PySRRegressor(
    niterations=50,
    binary_operators=binary_operators,
    unary_operators=unary_operators,
    extra_sympy_mappings=extra_sympy_mappings,
    constraints=constraints,
    nested_constraints=nested_constraints,
    select_k_features=n_features,
    maxsize=25,
    verbosity=1,
    parsimony=0.001,
)

model.fit(X, y)

### Construct a quadratic model for comparison
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Ensure that the matrix A in x^T A x is positive definite by adding a small value to the diagonal elements
epsilon = 1e-4
A = np.random.randn(X_poly.shape[1], X_poly.shape[1])
A = A.T @ A + epsilon * np.eye(X_poly.shape[1])
y_quad = np.einsum('ij,jk,ik->i', X_poly, A, X_poly)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

### Compare results with a positive definite quadratic model (x^T A x + b^T x + c)

y_pred_pysr = model.predict(X)
y_pred_quad = lin_reg.predict(X_poly)

mse_pysr = np.mean((y - y_pred_pysr) ** 2)
mse_quad = np.mean((y - y_pred_quad) ** 2)

print(f"Mean Squared Error (PySR model): {mse_pysr:.4f}")
print(f"Mean Squared Error (Quadratic model): {mse_quad:.4f}")


print("\nPySR Discovered Equations:")
print(model)

print("\nQuadratic Model Coefficients:")
print(f"Intercept: {lin_reg.intercept_}")
print(f"Coefficients: {lin_reg.coef_}")

# Optional: Plot the true vs predicted values for both models
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y, y_pred_pysr, alpha=0.5)
plt.xlabel('True Values')
plt.ylabel('PySR Predicted Values')
plt.title('PySR Model Predictions')

plt.subplot(1, 2, 2)
plt.scatter(y, y_pred_quad, alpha=0.5, color='orange')
plt.xlabel('True Values')
plt.ylabel('Quadratic Model Predicted Values')
plt.title('Quadratic Model Predictions')

plt.tight_layout()
plt.show()
