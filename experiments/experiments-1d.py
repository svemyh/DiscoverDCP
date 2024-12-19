import os
import shutil

import numpy as np
import cvxpy as cp
from pysr import PySRRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import sympy
from sympy import lambdify
from scipy.special import softmax, logsumexp

import jax.numpy as jnp
import jax
import torch



np.random.seed(1337) # set for reproducibility


# Good results
def hidden_model_absolute(X):
    """
    f(x) = |x0|
    """
    return np.abs(X[:, 0])

## 
def hidden_model_piecewise_exp_linear(X):
    """
    hidden_model_piecewise_exp_linear
    """
    condition = X[:, 1] > 0
    y = np.where(condition, np.exp(X[:, 0]), X[:, 2] + X[:, 3])
    return y

## 
def hidden_model_blended_max_exp(X):
    """
    f(x) = exp(x0 + max(x0, -5*x0) + x0^2) + 4* x0
    """
    return np.exp(X[:, 0] + np.maximum(X[:, 0], -5*X[:, 0]) + X[:, 0]**2) +  4* X[:, 0]

## 
def hidden_model_piecewise_composite(X):
    """
    f(x) = exp(x1) + x2^2 if x0 > 0 else -x1^2 + 3x3
    """
    condition = X[:, 0] > 0
    y = np.where(condition, np.exp(X[:, 1]) + X[:, 2]**2, -X[:, 1]**2 + 3 * X[:, 3])
    return y


# =========================================
# 1. Data Generation with Specified Mean and Standard Deviation
# =========================================
mu = 0
x_0 = np.array([0.0])
sigma = 1

n_samples = 1000
n_features = 1


MIN_X = -1
MAX_X = 1

# Sample data from N(mu, sigma^2)
X = x_0 + sigma * np.random.randn(n_samples, n_features)
X = np.random.uniform(MIN_X, MAX_X, size=(n_samples, n_features))




# =========================================
y = hidden_model_blended_max_exp(X)
# =========================================






y += 5.0* np.random.randn(n_samples) # Add some noise to the data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================
# 2. PySR model initialization, defining operators and model training
# =========================================
binary_operators = ["+", "*", "max"]
unary_operators = ["exp", "square"]

def sympy_max(x, y):
    return sympy.Max(x, y)

def sympy_abs(x):
    return sympy.Abs(x)

def sympy_square(x):
    return x**2

def hinge(x):
    return np.maximum(0, 1-x)

def softplus(x):
    return np.log1p(np.exp(x))

extra_sympy_mappings = {
    "square": sympy_square,
}

# template = TemplateExpressionSpec(
#     function_symbols=["f", "g"],
#     combine="((; f, g), (x0, x1)) -> f(x0) + g(x1)",
# )

# Set constraints to prevent multiplication between variables
constraints = {
    "exp": 5,
    "square": 5,
    "max": (-1, -1),
    "*": (1, 1),  # Both arguments must have complexity <= 1
}

nested_constraints = {
    "exp": {"exp": 0, "square": 1, "max": 0, "*": 1},
    "square": {"exp": 0, "square": 0, "max": 0, "*": 1},
    "max": {"exp": 0, "square": 0, "max": 0, "*": 0},
    "*": {"exp": 0, "square": 0, "max": 0, "*": 0},
}

MAX_COMPLEXITY = 50

pysr_model = PySRRegressor(
    niterations=50,
    binary_operators=binary_operators,
    unary_operators=unary_operators,
    extra_sympy_mappings=extra_sympy_mappings,
    constraints=constraints,
    nested_constraints=nested_constraints,
    select_k_features=n_features,
    maxsize=MAX_COMPLEXITY,
    verbosity=1,
    parsimony=0.001,
)
pysr_model.fit(X_train, y_train)

print("Done fitting PySR model.")
# =======================================================================
# Extracting both the discovered 'best' as well as most complex equation:
# =======================================================================

best_equation = pysr_model.get_best()

i = 0
while True:
    print(f"Trying to extract optimal equation at index {MAX_COMPLEXITY-i}...")
    try:
        optimal_equation = pysr_model.sympy(index=MAX_COMPLEXITY-i)
        print(f"Optimal Equation at index {MAX_COMPLEXITY-i}: {optimal_equation}")
        break
    except:
        i += 1
        continue


# =======================================================================
# Extracting alternative models using different libraries (PyTorch, JAX):
# =======================================================================
# jax_model = pysr_model.jax()
# print(f"JAX Model: {jax_model}")

# torch_model = pysr_model.pytorch()
# print(f"PyTorch Model: {torch_model}")


# =========================================
# 3. Quadratic Model Initialization and Training with Positive Semidefinite Constraint
# =========================================

# find lowest (approximate) epsilon that makes A positive semidefinite
eps = 1e-9
while True:
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Extract the number of polynomial features
    n_poly_features = X_train_poly.shape[1]

    # Define CVXPY variables for the quadratic model
    # We will define A as a symmetric positive semidefinite matrix directly
    A = cp.Variable((n_features, n_features))
    b = cp.Variable(n_features)
    c = cp.Variable()



    # Define the objective: minimize Mean Squared Error (MSE)
    objective = 0
    for i in range(X_train.shape[0]):
        xi = X_train[i]
        yi = y_train[i]
        # Compute quadratic term: x_i^T A x_i
        quadratic_term = cp.quad_form(xi, A)
        # Compute linear term: x_i^T b
        linear_term = xi @ b
        # Compute prediction: y_pred = x_i^T A x_i + x_i^T b + c
        y_pred = quadratic_term + linear_term + c
        # Accumulate squared error
        objective += cp.square(y_pred - yi)

    # Average the MSE over the training samples
    objective = cp.Minimize(objective / X_train.shape[0])   

    # Define constraints to ensure A is symmetric and positive semidefinite
    constraints = [
        A >>  eps * np.eye(n_features),  # Tightened PSD constraint
        A == A.T  # Ensuring symmetry
    ]

    if eps >= 1e-4:
        A_opt = np.zeros((n_features, n_features))
        b_opt = np.zeros(n_features)
        c_opt = 0
        break

    # Define and solve the optimization problem using CVXOPT for better precision
    try:
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CVXOPT, verbose=True)
    except:
        # Fallback to SCS if CVXOPT fails
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=True)

    # Check if the problem was solved successfully
    print(f"\nOptimization Status for epsilon = {eps:.1e}: {prob.status}")
    if prob.status not in ["infeasible", "unbounded"]:
        # Extract the coefficients
        A_opt = A.value
        b_opt = b.value
        c_opt = c.value

        # Compute and print eigenvalues of A_opt
        eigenvalues = np.linalg.eigvalsh(A_opt)
        print("\nEigenvalues of Quadratic Coefficient Matrix (A):")
        for idx, eig in enumerate(eigenvalues):
            print(f"Eigenvalue {idx + 1}: {eig:.6f}")

        if np.all(eigenvalues >= -eps):
            print(f"\PSD Constraint Satisfied with epsilon = {eps:.1e}")
            break
        else:
            eps *= 10
    else:
        eps *= 10
        print("The optimization problem did not solve successfully for epsilon = {eps:.1e}")


    


# =========================================
# 4. Predictions on Test Set
# =========================================
# PySR model 'best_equation' predictions on test set
y_pred_pysr = pysr_model.predict(X_test)


####### using equation with highest complexity score #######
variables = [f'x{i}' for i in range(X_test.shape[1])]
optimal_equation_func = lambdify(variables, optimal_equation, modules="numpy")
y_pred_pysr_opt = np.array([optimal_equation_func(*x) for x in X_test])



# Quadratic model predictions on test set using the optimized coefficients
# y_pred_quad = x^T A x + b^T x + c
y_pred_quad = np.array([
    xi @ A_opt @ xi + xi @ b_opt + c_opt for xi in X_test
])

# =========================================
# 5. Error Metrics Calculation
# =========================================
# Calculate error metrics on test set
mse_pysr = mean_squared_error(y_test, y_pred_pysr)
mse_pysr_opt = mean_squared_error(y_test, y_pred_pysr_opt)
mse_quad = mean_squared_error(y_test, y_pred_quad)
mae_pysr = mean_absolute_error(y_test, y_pred_pysr)
mae_pysr_opt = mean_absolute_error(y_test, y_pred_pysr_opt)
mae_quad = mean_absolute_error(y_test, y_pred_quad)

print(f"Mean Squared Error (PySR model) on Test Set: {mse_pysr:.4f}")
print(f"Mean Squared Error (PySR_opt model) on Test Set: {mse_pysr_opt:.4f}")
print(f"Mean Squared Error (Quadratic model) on Test Set: {mse_quad:.4f}")
print(f"Mean Absolute Error (PySR model) on Test Set: {mae_pysr:.4f}")
print(f"Mean Absolute Error (PySR_opt model) on Test Set: {mae_pysr_opt:.4f}")
print(f"Mean Absolute Error (Quadratic model) on Test Set: {mae_quad:.4f}\n")

print("PySR Discovered Equations:")
print(pysr_model)

# =========================================
# 6. Quadratic Model in Matrix Form with Numerical Values
# =========================================
print("\nQuadratic Model Equation:")
# Get feature names
feature_names = poly.get_feature_names_out()

# Build the quadratic equation string: y = x^T A x + b^T x + c
quadratic_equation = f"y = {c_opt:.4f}"
for i in range(n_features):
    for j in range(n_features):
        if A_opt[i, j] != 0:
            if i == j:
                quadratic_equation += f" + {A_opt[i, j]:.4f}*x{i}^2"
            elif i < j:
                quadratic_equation += f" + {A_opt[i, j]:.4f}*x{i}*x{j}"
# Add linear terms
linear_terms = " + " + " + ".join([f"{b_opt[i]:.4f}*x{i}" for i in range(n_features)]) if any(b_opt) else ""
quadratic_equation += linear_terms
print(quadratic_equation)

print("\nQuadratic Coefficient Matrix (A):")
print(A_opt)

print("\nLinear Coefficients (b):")
print(b_opt)

# =========================================
# 7. Visualizations
# =========================================

# -----------------------------------------
# Figure 1: True vs Predicted Values for All Models
# -----------------------------------------
plt.figure(figsize=(21, 6))

# Subplot 1: PySR Models Predictions vs True Values
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_pysr, alpha=0.5, label='PySR', color='blue')
plt.scatter(y_test, y_pred_pysr_opt, alpha=0.5, label='PySR_opt', color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('PySR Models Predictions vs True Values')
plt.legend()

# Subplot 2: Quadratic Model Predictions vs True Values
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_quad, alpha=0.5, label='Quadratic', color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Quadratic Model Predictions vs True Values')
plt.legend()

# Subplot 3: All Models Predictions vs True Values
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_pysr, alpha=0.3, label='PySR', color='blue')
plt.scatter(y_test, y_pred_pysr_opt, alpha=0.3, label='PySR_opt', color='purple')
plt.scatter(y_test, y_pred_quad, alpha=0.3, label='Quadratic', color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('All Models Predictions vs True Values')
plt.legend()

plt.tight_layout()
plt.show()

# -----------------------------------------
# Figure 2: Residuals Distribution and Error Metrics
# -----------------------------------------
plt.figure(figsize=(21, 6))

# Subplot 1: Residuals Distribution
plt.subplot(1, 3, 1)
residuals_pysr = y_test - y_pred_pysr
residuals_pysr_opt = y_test - y_pred_pysr_opt
residuals_quad = y_test - y_pred_quad

plt.hist(residuals_pysr, bins=30, alpha=0.3, label='PySR', color='blue', density=True)
plt.hist(residuals_pysr_opt, bins=30, alpha=0.3, label='PySR_opt', color='purple', density=True)
plt.hist(residuals_quad, bins=30, alpha=0.3, label='Quadratic', color='orange', density=True)
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.title('Residuals Distribution on Test Set')
plt.legend()

# Subplot 2: Error Metrics Comparison (MSE and MAE)
plt.subplot(1, 3, 2)
metrics = ['MSE', 'MAE']
pysr_metrics = [mse_pysr, mae_pysr]
pysr_opt_metrics = [mse_pysr_opt, mae_pysr_opt]
quad_metrics = [mse_quad, mae_quad]

x = np.arange(len(metrics))  # the label locations
width = 0.2  # the width of the bars

plt.bar(x - 1.5*width, pysr_metrics, width, label='PySR', color='blue')
plt.bar(x + 0.5*width, pysr_opt_metrics, width, label='PySR_opt', color='purple')
plt.bar(x + 1.5*width, quad_metrics, width, label='Quadratic', color='orange')

plt.ylabel('Error')
plt.title('Error Metrics Comparison on Test Set')
plt.xticks(x, metrics)
plt.legend()

# Subplot 3: Combined Error Metrics Visualization
plt.subplot(1, 3, 3)
width = 0.2
plt.bar(x - 1.5*width, pysr_metrics, width, label='PySR', color='blue')
plt.bar(x + 0.5*width, pysr_opt_metrics, width, label='PySR_opt', color='purple')
plt.bar(x + 1.5*width, quad_metrics, width, label='Quadratic', color='orange')
plt.ylabel('Error')
plt.title('All Models Error Metrics')
plt.xticks(x, metrics)
plt.legend()

plt.tight_layout()
plt.show()

# -----------------------------------------
# Figure 3: Distribution of True vs Predicted Values for All Models
# -----------------------------------------
plt.figure(figsize=(21, 6))

# Subplot 1: True vs PySR Models Predicted Values Distribution
plt.subplot(1, 3, 1)
plt.hist(y_test, bins=30, alpha=0.5, label='True Values', color='green', density=True)
plt.hist(y_pred_pysr, bins=30, alpha=0.5, label='PySR', color='blue', density=True)
plt.hist(y_pred_pysr_opt, bins=30, alpha=0.5, label='PySR_opt', color='purple', density=True)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('True vs PySR Models Predicted Values Distribution')
plt.legend()

# Subplot 2: True vs Quadratic Model Predicted Values Distribution
plt.subplot(1, 3, 2)
plt.hist(y_test, bins=30, alpha=0.5, label='True Values', color='green', density=True)
plt.hist(y_pred_quad, bins=30, alpha=0.5, label='Quadratic', color='orange', density=True)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('True vs Quadratic Predicted Values Distribution')
plt.legend()

# Subplot 3: Combined Distribution for All Models
plt.subplot(1, 3, 3)
plt.hist(y_test, bins=30, alpha=0.3, label='True Values', color='green', density=True)
plt.hist(y_pred_pysr, bins=30, alpha=0.3, label='PySR', color='blue', density=True)
plt.hist(y_pred_pysr_opt, bins=30, alpha=0.3, label='PySR_opt', color='purple', density=True)
plt.hist(y_pred_quad, bins=30, alpha=0.3, label='Quadratic', color='orange', density=True)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('All Models vs True Values Distribution')
plt.legend()

plt.tight_layout()
plt.show()

# -----------------------------------------
# Figure 4: Quadratic Coefficient Matrix Heatmap
# -----------------------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(A_opt, annot=True, fmt=".4f", cmap='viridis',
            xticklabels=[f"x{i}" for i in range(n_features)],
            yticklabels=[f"x{i}" for i in range(n_features)])
plt.title('Quadratic Coefficient Matrix (A) Heatmap')
plt.xlabel('Feature')
plt.ylabel('Feature')
plt.show()

# =========================================
# 8. Eigenvalues and Residual Statistics
# =========================================

# Compute and print eigenvalues of A_opt
eigenvalues = np.linalg.eigvalsh(A_opt)  # More efficient for symmetric matrices
print("\nEigenvalues of Quadratic Coefficient Matrix (A):")
for idx, eig in enumerate(eigenvalues):
    print(f"Eigenvalue {idx + 1}: {eig:.6f}")

print(f"\n'Best' Discovered Equation by PySR: {best_equation.sympy_format}")

print(f"\n'Optimal' Discovered Equation by PySR: {optimal_equation}")



# After the run, move 'hall_of_fame'-files to 'hall_of_fame/' directory
for filename in os.listdir('.'):
    if filename.startswith('hall_of_fame') and (filename.endswith('.csv') or filename.endswith('.pkl') or filename.endswith('.csv.bkup')):
        shutil.move(filename, os.path.join('hall_of_fame', filename))
