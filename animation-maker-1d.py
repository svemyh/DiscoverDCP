import logging
import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imageio

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Hidden model
def hidden_model_absolute(X):
    """
    f(x) = |x0|
    """
    return np.abs(X[:, 0])

def hidden_model_blended_max_exp(X):
    """
    f(x) = exp(x0 + max(x0, -5*x0) + x0^2) + 4* x0
    """
    return np.exp(X[:, 0] + np.maximum(X[:, 0], -5*X[:, 0]) + X[:, 0]**2) +  4* X[:, 0]


# Domain for plotting

MIN_X = -1
MAX_X = 1

x_plot = np.linspace(MIN_X, MAX_X, 400)
X_plot = x_plot.reshape(-1, 1)
y_true = hidden_model_blended_max_exp(X_plot)


### Generate data from the hidden model
n_samples = 150
n_features = 1
X = np.random.uniform(MIN_X, MAX_X, size=(n_samples, n_features))
y_sample = hidden_model_blended_max_exp(X)
y_sample += 5.0* np.random.randn(n_samples) # Add some noise to the data



# Read data from CSV
df = pd.read_csv('hall_of_fame/hall_of_fame_2024-12-18_015836.806.csv')

# Define safe_eval function to handle the given equations
def square(z):
    return z*z

def safe_eval(expr, x):
    # Evaluate the given expression string for array x
    # Replaces variable x0 with x
    # Allowed functions: max, square
    logger.debug(f"safe_eval: expr: {expr}")
    logger.debug(f"safe_eval: x: {x}")

    y = np.zeros_like(x)

    local_dict = {
        'max': np.maximum,  # vectorized max
        'square': square,
        'exp': np.exp,
    }

    # Evaluate each element separately to handle max(...) calls correctly
    for i in range(len(x)):
        #logger.debug(f"safe_eval: i: {i}")
        #logger.debug(f"safe_eval: x[i]: {x[i]}")
        local_dict['x0'] = x[i]
        y[i] = eval(expr, {"__builtins__": None}, local_dict)

    return y

# Create directory for frames
if not os.path.exists("tmp_frames"):
    os.makedirs("tmp_frames")

# Generate a timestamp for the final GIF filename
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

filenames = []
for i, row in df.iterrows():
    complexity = row['Complexity']
    logger.debug(f"Complexity: {complexity}")
    loss = row['Loss']
    eq = row['Equation']
    
    # Compute approximation
    logger.debug(f"Computing approximation for equation: {eq}")
    y_approx = safe_eval(eq, x_plot)
    logger.debug(f"y_approx: {y_approx}")

    # Plot
    plt.figure(figsize=(6,4))
    # Plot hidden model
    plt.plot(x_plot, y_true, 'r-', label="Hidden Model", linewidth=2)
    # Plot approximation
    plt.plot(x_plot, y_approx, 'b-', label=f"DiscoverDCP", linewidth=2)
    # Plot benchmark convex quadratic model
    y_alternative = -4.9182 + 61.5890 * x_plot**2 -20.3866 * x_plot
    plt.plot(x_plot, y_alternative, 'y-', label="Quadratic (L2 minimizer)", linewidth=2)

    plt.scatter(X, y_sample, color='black', label="Data sampled from Hidden Model", s=10)

    # Set title and labels
    plt.title(f"Complexity: {complexity}, Loss: {loss:.4g}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(MIN_X - abs(MAX_X-MIN_X)*0.05, MAX_X + abs(MAX_X-MIN_X)*0.05)
    plt.ylim(np.min(y_true) - abs(np.min(y_true)-np.max(y_true))*0.05, np.max(y_true) + abs(np.min(y_true)-np.max(y_true))*0.05)
    #plt.xlim(MIN_X, MAX_X)
    #plt.ylim(np.min(y_true), np.max(y_true))

    # Add the equation text
    #plt.text(MIN_X + abs(MAX_X-MIN_X)*0.02, np.min(y_true) + abs(np.min(y_true)-np.max(y_true))*0.05, f"Equation: {eq}", fontsize=7, wrap=True)

    plt.legend(loc="upper right")
    plt.tight_layout()
    
    frame_filename = f"tmp_frames/frame_{i:03d}.png"
    plt.savefig(frame_filename, dpi=150)
    plt.close()
    filenames.append(frame_filename)

# Create directory for animations
if not os.path.exists("animations"):
    os.makedirs("animations")

# Create and save GIF with infinite loop and longer duration
gif_filename = f"animations/approximation_progress_{timestamp}.gif"
with imageio.get_writer(gif_filename, mode='I', duration=1200.0, loop=0) as writer:
    for f in filenames:
        image = imageio.v2.imread(f)
        writer.append_data(image)

print(f"GIF saved as {gif_filename}")
