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


np.random.seed(1337) # set for reproducibility


class PiecewiseLinearConvex:
    def __init__(self, n_features=4, n_pieces=5, smoothing_factor=0.1):
        """
        Initialize piecewise linear model with near-convex properties
        
        Args:
            n_features: Number of input features
            n_pieces: Number of linear pieces
            smoothing_factor: Controls smoothness at transition points
        """
        self.n_features = n_features
        self.n_pieces = n_pieces
        self.smoothing_factor = smoothing_factor
        
        # Generate pieces that are almost convex globally
        self.weights, self.biases, self.boundaries = self._initialize_pieces()
        
    def _initialize_pieces(self):
        """
        Initialize piece parameters to maintain approximate global convexity
        """
        # Create base matrix for weights with increasing slopes
        base_weights = np.linspace(-1, 1, self.n_pieces)
        weights = np.zeros((self.n_pieces, self.n_features))
        
        # Make each piece's weights slightly different but maintaining rough convexity
        for i in range(self.n_pieces):
            piece_weights = base_weights[i] + 0.1 * np.random.randn(self.n_features)
            # Ensure weights increase slightly for approximate convexity
            weights[i] = piece_weights + i * 0.1
            
        # Generate biases to ensure continuity at boundaries
        biases = np.zeros(self.n_pieces)
        for i in range(1, self.n_pieces):
            # Adjust bias to maintain continuity
            biases[i] = biases[i-1] + np.random.uniform(0, 0.2)
            
        # Create boundaries that partition the input space
        boundaries = np.linspace(-2, 2, self.n_pieces + 1)
        
        return weights, biases, boundaries
    
    def _smooth_max(self, x, alpha=10.0):
        """
        Smooth maximum approximation using log-sum-exp
        """
        return (1/alpha) * np.log(np.sum(np.exp(alpha * x), axis=-1))
    
    def _evaluate_piece(self, X, piece_idx):
        """
        Evaluate a single linear piece
        """
        return np.dot(X, self.weights[piece_idx]) + self.biases[piece_idx]
    
    def _smooth_indicator(self, x, boundary, smoothing):
        """
        Smooth indicator function for transitions between pieces
        """
        return 1 / (1 + np.exp(-(x - boundary)/smoothing))
    
    def __call__(self, X):
        """
        Evaluate the piecewise linear function with smooth transitions
        """
        batch_size = len(X)
        # Compute norm of input for piece selection
        X_norm = np.linalg.norm(X, axis=1)
        
        # Initialize output array
        outputs = np.zeros(batch_size)
        
        # Evaluate each piece and combine with smooth transitions
        for i in range(self.n_pieces):
            lower_bound = self.boundaries[i]
            upper_bound = self.boundaries[i + 1]
            
            # Compute piece contribution
            piece_output = self._evaluate_piece(X, i)
            
            # Smooth transition weights
            lower_weight = self._smooth_indicator(X_norm, lower_bound, self.smoothing_factor)
            upper_weight = 1 - self._smooth_indicator(X_norm, upper_bound, self.smoothing_factor)
            
            # Combine weights and add to output
            piece_weight = lower_weight * upper_weight
            outputs += piece_output * piece_weight
            
        # Add small quadratic regularization term for strict convexity
        outputs += 0.01 * np.sum(X**2, axis=1)
        
        return outputs


def verify_approximate_convexity(model, n_samples=1000, n_features=4, tolerance=1e-3):
    """
    Verify the approximate convexity of the model
    
    Returns:
        float: Proportion of points satisfying convexity condition within tolerance
    """
    # Generate random points for testing
    X1 = np.random.randn(n_samples, n_features)
    X2 = np.random.randn(n_samples, n_features)
    
    # Generate random convex combinations
    t = np.random.rand(n_samples, 1)
    X_combined = t * X1 + (1 - t) * X2
    
    # Evaluate function at these points
    y1 = model(X1)
    y2 = model(X2)
    y_combined = model(X_combined)
    
    # Check convexity condition with tolerance
    convexity_violations = y_combined - (t.flatten() * y1 + (1 - t).flatten() * y2)
    
    # Count approximately satisfying points
    n_satisfying = np.sum(convexity_violations <= tolerance)
    
    return n_satisfying / n_samples


def hidden_model(X):
    """
    Hidden function to be discovered: f(x) = exp(x0 + x1) + (x2)^2 + 3*x3 + 2
    """
    return np.exp(X[:, 0] + X[:, 1]) + np.square(X[:, 2]) + 3 * X[:, 3] + 2

def hidden_model_logistic(X):
    """
    Hidden logistic function to be discovered.
    f(x) = 1 / (1 + exp(-(w.T @ X + b)))
    """
    # Example coefficients
    w = np.array([0.5, -1.2, 0.8, 0.3])
    b = -0.7
    return 1 / (1 + np.exp(-(X @ w + b)))


def hidden_model_harmonic(X):
    """
    Hidden harmonic oscillator potential to be discovered.
    f(x) = 0.5 * k * (x - x0)^2 + c
    """
    # Example coefficients
    k = 2.5
    x0 = np.array([1.0, -0.5, 0.8, 0.0])  # Equilibrium positions
    c = 1.5
    return 0.5 * k * np.sum((X - x0) ** 2, axis=1) + c

# 1. Linear Model (Basic, Convex)
def hidden_model_linear(X):
    """
    f(x) = w.T @ X + b
    """
    w = np.array([1.2, -0.7, 0.5, 0.9])
    b = 0.3
    return X @ w + b

# 2. Polynomial Model (Basic, Convex for specific constraints)
def hidden_model_polynomial(X):
    """
    f(x) = sum(a_i * x_i^2) + sum(b_i * x_i) + c
    """
    a = np.array([0.5, 1.0, -0.8, 0.3])
    b = np.array([0.7, -1.2, 0.4, 0.6])
    c = 1.0
    return np.sum(a * X**2, axis=1) + np.sum(b * X, axis=1) + c

# 3. Exponential Decay (Convex)
def hidden_model_exponential(X):
    """
    f(x) = exp(-w.T @ X)
    """
    w = np.array([1.0, -0.5, 0.3, 0.2])
    return np.exp(-X @ w)

# 4. Rosenbrock Function (Non-convex)
def hidden_model_rosenbrock(X):
    """
    f(x) = sum(100 * (x_i+1 - x_i^2)^2 + (1 - x_i)^2)
    """
    return np.sum(100 * (X[:, 1:] - X[:, :-1]**2)**2 + (1 - X[:, :-1])**2, axis=1)

# 5. Sinusoidal Oscillation (Non-convex)
def hidden_model_sinusoidal(X):
    """
    f(x) = sin(w.T @ X + b)
    """
    w = np.array([0.8, -1.5, 0.7, 0.3])
    b = 0.5
    return np.sin(X @ w + b)

# 6. Quadratic with Mixed Terms (Convex)
def hidden_model_quadratic(X):
    """
    f(x) = 0.5 * x.T @ A @ x + b.T @ x + c
    """
    A = np.array([[2.0, 0.5, 0.0, 0.0],
                  [0.5, 1.5, 0.2, 0.0],
                  [0.0, 0.2, 3.0, 0.1],
                  [0.0, 0.0, 0.1, 1.0]])
    b = np.array([0.5, -0.3, 0.2, 0.4])
    c = 1.2
    return np.einsum('ij,ij->i', X @ A, X) + X @ b + c

# 7. Logistic Model (Non-convex)
def hidden_model_logistic(X):
    """
    f(x) = 1 / (1 + exp(-(w.T @ X + b)))
    """
    w = np.array([0.5, -1.2, 0.8, 0.3])
    b = -0.7
    return 1 / (1 + np.exp(-(X @ w + b)))

# 8. Gaussian Model (Convex)
def hidden_model_gaussian(X):
    """
    f(x) = exp(-||x - mu||^2 / (2 * sigma^2))
    """
    mu = np.array([1.0, 0.5, -0.5, 0.0])
    sigma = 1.0
    return np.exp(-np.sum((X - mu)**2, axis=1) / (2 * sigma**2))

# Additional Hidden Models

def hidden_model_rational(X):
    """
    Rational function model: f(x) = (ax + b)/(cx + d)
    Good test case for discovering fractional relationships
    """
    a = np.array([0.5, 1.2, -0.3, 0.8])
    b = 2.0
    c = np.array([0.2, -0.4, 0.1, 0.3])
    d = 1.0
    numerator = X @ a + b
    denominator = X @ c + d
    return numerator / denominator

def hidden_model_periodic_mixture(X):
    """
    Mixture of periodic functions: f(x) = sin(wx) + cos(vx)
    Tests ability to discover periodic components
    """
    w = np.array([1.2, -0.5, 0.8, 0.3])
    v = np.array([0.6, 0.9, -0.4, 0.2])
    return np.sin(X @ w) + np.cos(X @ v)

def hidden_model_piecewise(X):
    """
    Piecewise function with different behavior in different regions
    Tests ability to discover conditional relationships
    """
    conditions = X[:, 0] > 0
    y = np.zeros(len(X))
    y[conditions] = np.exp(X[conditions, 1]) + X[conditions, 2]**2
    y[~conditions] = -X[~conditions, 1]**2 + 3*X[~conditions, 3]
    return y


def hidden_model_log_sum_exp(X):
    """
    Log-sum-exp function (smooth maximum approximation)
    Strictly convex and differentiable
    f(x) = log(sum(exp(w_i * x_i + b_i))) + quadratic regularization
    """
    w = np.array([[1.2, -0.7, 0.5, 0.9],
                  [-0.8, 1.1, -0.4, 0.6],
                  [0.5, -0.9, 1.3, -0.7]])
    b = np.array([0.3, -0.2, 0.4])
    
    # Compute log-sum-exp term
    activations = np.dot(X, w.T) + b
    lse_term = logsumexp(activations, axis=1)
    
    # Add quadratic regularization to ensure strong convexity
    quad_term = 0.1 * np.sum(X**2, axis=1)
    
    return lse_term + quad_term

def hidden_model_entropic(X):
    """
    Entropy-regularized linear model with barrier terms
    Strictly convex due to the entropy term
    f(x) = w^T x - entropy(softmax(Ax)) - sum(log(-x_i + upper_bound))
    """
    w = np.array([0.8, -0.5, 0.3, 0.6])
    A = np.array([[1.2, -0.7, 0.4, -0.3],
                  [-0.5, 0.9, -0.6, 0.4],
                  [0.3, -0.4, 1.1, -0.8]])
    
    # Linear term
    linear = np.dot(X, w)
    
    # Entropy term using softmax
    logits = np.dot(X, A.T)
    probs = softmax(logits, axis=1)
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    
    # Barrier terms (log barriers for box constraints)
    upper_bound = 5.0
    barrier = np.sum(np.log(upper_bound - X + 1e-6), axis=1)
    
    return linear - entropy + 0.1 * barrier

def hidden_model_huber_complex(X):
    """
    Complex model using Huber-like smoothing of various terms
    Nearly convex due to careful parameter selection
    f(x) = smooth_huber(quad) + log_barrier(linear) + prox_term
    """
    def smooth_huber(x, delta=1.0):
        mask = np.abs(x) <= delta
        return np.where(mask,
                       0.5 * x**2,
                       delta * np.abs(x) - 0.5 * delta**2)
    
    A = np.array([[2.0, 0.5, 0.3, 0.1],
                  [0.5, 1.5, 0.2, 0.4],
                  [0.3, 0.2, 1.0, 0.3],
                  [0.1, 0.4, 0.3, 1.2]])
    
    b = np.array([0.5, -0.3, 0.2, 0.4])
    
    # Quadratic term with Huber smoothing
    quad_term = np.dot(X, np.dot(A, X.T).T)
    huber_term = np.sum(smooth_huber(quad_term), axis=1)
    
    # Log-barrier term for linear constraints
    linear_term = np.dot(X, b)
    barrier_term = -np.log(10.0 - np.abs(linear_term) + 1e-6)
    
    # Proximal regularization
    prox_term = 0.1 * np.sum(X**2, axis=1)
    
    return huber_term + barrier_term + prox_term

def hidden_model_self_concordant(X):
    """
    Self-concordant barrier function with quadratic mixing
    Convex by construction through self-concordant properties
    f(x) = -sum(log(u_i - l_i)) + quad_form + log_sum_exp
    """
    # Parameters for the barrier terms
    upper_bounds = np.array([5.0, 4.0, 4.5, 3.5])
    lower_bounds = np.array([-5.0, -4.0, -4.5, -3.5])
    
    # Quadratic form matrix (positive definite)
    Q = np.array([[1.0, 0.3, 0.1, 0.0],
                  [0.3, 1.2, 0.2, 0.1],
                  [0.1, 0.2, 1.5, 0.3],
                  [0.0, 0.1, 0.3, 1.8]])
    
    # Barrier terms
    upper_barriers = np.sum(np.log(upper_bounds - X + 1e-6), axis=1)
    lower_barriers = np.sum(np.log(X - lower_bounds + 1e-6), axis=1)
    
    # Quadratic term
    quad_term = np.sum(X * np.dot(X, Q), axis=1)
    
    # Log-sum-exp term for smoothing
    lse_term = logsumexp(X, axis=1)
    
    return -upper_barriers - lower_barriers + 0.5 * quad_term + 0.1 * lse_term

def hidden_model_moreau_yosida(X):
    """
    Moreau-Yosida regularization of a complex convex function
    Provides better smoothness properties while maintaining convexity
    f(x) = h(x) + (1/2λ)||x - prox_λh(x)||^2
    """
    def prox_operator(X, lambda_param=1.0):
        """Proximal operator approximation"""
        Q = np.array([[1.5, 0.4, 0.1, 0.0],
                     [0.4, 2.0, 0.3, 0.1],
                     [0.1, 0.3, 1.8, 0.2],
                     [0.0, 0.1, 0.2, 1.6]])
        
        return X / (1 + lambda_param * np.dot(Q, X.T).T)
    
    # Parameters
    lambda_reg = 0.5
    w = np.array([0.7, -0.4, 0.3, 0.5])
    
    # Base convex function h(x)
    h_x = np.dot(X, w) + 0.5 * np.sum(X**2, axis=1)
    
    # Proximal mapping
    prox_x = prox_operator(X, lambda_reg)
    
    # Moreau-Yosida regularization term
    moreau_term = (1.0 / (2 * lambda_reg)) * np.sum((X - prox_x)**2, axis=1)
    
    return h_x + moreau_term

def test_convexity(model, n_points=1000, n_tests=100):
    """
    Test approximate convexity of a model through random sampling
    Returns: proportion of tests that satisfy convexity condition
    """
    n_features = 4
    convex_tests = 0
    
    for _ in range(n_tests):
        # Generate random points for convexity test
        x1 = np.random.randn(n_points, n_features)
        x2 = np.random.randn(n_points, n_features)
        t = np.random.rand(n_points, 1)  # Random weights for convex combination
        
        # Compute convex combination
        x_combined = t * x1 + (1-t) * x2
        
        # Evaluate function at these points
        y1 = model(x1)
        y2 = model(x2)
        y_combined = model(x_combined)
        
        # Check if convexity condition holds
        if np.all(y_combined <= t.flatten() * y1 + (1-t).flatten() * y2):
            convex_tests += 1
    
    return convex_tests / n_tests


# Good results
def hidden_model_absolute(X):
    """
    f(x) = |x0| + |x1| + |x2| + |x3|
    """
    return np.abs(X[:, 0]) + np.abs(X[:, 1]) + np.abs(X[:, 2]) + np.abs(X[:, 3])


def hidden_model_max_min(X):
    """
    f(x) = max(x0, x1) + min(x2, x3)
    """
    return np.maximum(X[:, 0], X[:, 1]) + np.minimum(X[:, 2], X[:, 3])



## LEts goooo!!!
def hidden_model_piecewise_exp_linear(X):
    """
    hidden_model_piecewise_exp_linear
    """
    condition = X[:, 1] > 0
    y = np.where(condition, np.exp(X[:, 0]), X[:, 2] + X[:, 3])
    return y


def hidden_model_softplus(X):
    """
    f(x) = log(1 + exp(x0)) + log(1 + exp(x1)) + x2 + x3
    """
    return np.log1p(np.exp(X[:, 0])) + np.log1p(np.exp(X[:, 1])) + X[:, 2] + X[:, 3]

## LEts goooo!!!

def hidden_model_blended_max_exp(X):
    """
    f(x) = exp(x0 + max(x1, x2)) + x3^2
    """
    return np.exp(X[:, 0] + np.maximum(X[:, 1], X[:, 2])) + X[:, 3]**2


def hidden_model_log_linear(X):
    """
    f(x) = log(1 + exp(x0 + x1)) + x2^2 + x3
    """
    return np.log1p(np.exp(X[:, 0] + X[:, 1])) + X[:, 2]**2 + X[:, 3]


def hidden_model_multiplicative_interaction(X):
    """
    f(x) = x0 * x1 + x2^2 + x3
    """
    return X[:, 0] * X[:, 1] + X[:, 2]**2 + X[:, 3]


def hidden_model_log_sum_exp_combined(X):
    """
    f(x) = log(exp(x0) + exp(x1)) + x2 + x3
    """
    return np.log(np.exp(X[:, 0]) + np.exp(X[:, 1])) + X[:, 2] + X[:, 3]


def hidden_model_rational_2(X):
    """
    Rational function model: f(x) = (0.5x0 + 1.2x1 - 0.3x2 + 0.8x3 + 2.0) / (0.2x0 - 0.4x1 + 0.1x2 + 0.3x3 + 1.0)
    """
    numerator = 0.5 * X[:, 0] + 1.2 * X[:, 1] - 0.3 * X[:, 2] + 0.8 * X[:, 3] + 2.0
    denominator = 0.2 * X[:, 0] - 0.4 * X[:, 1] + 0.1 * X[:, 2] + 0.3 * X[:, 3] + 1.0
    return numerator / denominator


def hidden_model_periodic_mixture(X):
    """
    f(x) = sin(1.2x0 - 0.5x1 + 0.8x2 + 0.3x3) + cos(0.6x0 + 0.9x1 - 0.4x2 + 0.2x3)
    """
    term1 = np.sin(1.2 * X[:, 0] - 0.5 * X[:, 1] + 0.8 * X[:, 2] + 0.3 * X[:, 3])
    term2 = np.cos(0.6 * X[:, 0] + 0.9 * X[:, 1] - 0.4 * X[:, 2] + 0.2 * X[:, 3])
    return term1 + term2


## almost there
def hidden_model_piecewise_composite(X):
    """
    f(x) = exp(x1) + x2^2 if x0 > 0 else -x1^2 + 3x3
    """
    condition = X[:, 0] > 0
    y = np.where(condition, np.exp(X[:, 1]) + X[:, 2]**2, -X[:, 1]**2 + 3 * X[:, 3])
    return y


def hidden_model_log_sum_exp_quadratic(X):
    """
    f(x) = log(sum(exp(w_i * x + b_i))) + 0.1 * ||x||^2
    """
    w = np.array([[1.2, -0.7, 0.5, 0.9],
                  [-0.8, 1.1, -0.4, 0.6],
                  [0.5, -0.9, 1.3, -0.7]])
    b = np.array([0.3, -0.2, 0.4])
    
    activations = np.dot(X, w.T) + b
    lse_term = logsumexp(activations, axis=1)
    quad_term = 0.1 * np.sum(X**2, axis=1)
    
    return lse_term + quad_term





# =========================================
# 1. Data Generation with Specified Mean and Standard Deviation
# =========================================
mu = 1
x_0 = np.array([-1.0, -1.0, -1.0, -1.0])
sigma = 1

n_samples = 10000
n_features = 4

# Sample data from N(mu, sigma^2)
X = x_0 + sigma * np.random.randn(n_samples, n_features)





# =========================================
y = hidden_model_quadratic(X)
# =========================================






y += 1.0* np.random.randn(n_samples) # Add some noise to the data

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

extra_sympy_mappings = {
    "max": sympy_max,
    "square": lambda x: x**2,
}

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

MAX_COMPLEXITY = 200

pysr_model = PySRRegressor(
    niterations=5000,
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

# =========================================
# Print the Best-Discovered Equation by PySR
# =========================================

print("pysr_model: ", pysr_model)
#print("pysr_model[1]: ", pysr_model[1])
print("pysr_model.get_best().sympy_format: ", pysr_model.get_best().sympy_format)
#print("pysr_model(1).sympy_format: ", pysr_model(1).sympy_format)
#print("pysr_model.index(1).sympy_format: ", pysr_model.index(1).sympy_format)


i = 0
while True:
    try:
        optimal_equation = pysr_model.sympy(index=MAX_COMPLEXITY-i)
        print(f"Optimal Equation at index {MAX_COMPLEXITY-i}: {optimal_equation}")
        break
    except:
        i += 1
        continue



best_equation = pysr_model.get_best()

print(f"\nBest Discovered Equation by PySR: {best_equation.sympy_format}")


# =========================================
# 3. Quadratic Model Initialization and Training with Positive Semidefinite Constraint
# =========================================
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


# find lowest (approximate) epsilon that makes A positive semidefinite
eps = 1e-9
while True:

    # Define constraints to ensure A is symmetric and positive semidefinite
    constraints = [
        A >>  eps * np.eye(n_features),  # Tightened PSD constraint
        A == A.T  # Ensuring symmetry
    ]

    # Define and solve the optimization problem using CVXOPT for better precision
    try:
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CVXOPT, verbose=True)
    except:
        # Fallback to SCS if CVXOPT fails
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=True)

    # Check if the problem was solved successfully
    if prob.status not in ["infeasible", "unbounded"]:
        # Extract the coefficients
        A_opt = A.value
        b_opt = b.value
        c_opt = c.value
    else:
        raise ValueError("The optimization problem did not solve successfully.")


    # Compute and print eigenvalues of A_opt
    eigenvalues = np.linalg.eigvalsh(A_opt)  # More efficient for symmetric matrices
    print("\nEigenvalues of Quadratic Coefficient Matrix (A):")
    for idx, eig in enumerate(eigenvalues):
        print(f"Eigenvalue {idx + 1}: {eig:.6f}")

    if np.all(eigenvalues >= -eps):
        print(f"\PSD Constraint Satisfied with epsilon = {eps:.1e}")
        break
    else:
        eps *= 10




# =========================================
# 4. Predictions on Test Set
# =========================================
# PySR model predictions on test set
y_pred_pysr = pysr_model.predict(X_test)





####### using actual optimal equation #######
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
