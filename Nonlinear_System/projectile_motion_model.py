"""
Project:
Comparative Machine Learning Modeling of Linear and Nonlinear Physical Systems

Nonlinear Physical System:
Projectile Motion using TensorFlow

Author: Ujala
Department of Physics
Baba Guru Nanak University, Nankana Sahib
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# -------------------------------------------------
# Generate Projectile Motion Dataset
# -------------------------------------------------

np.random.seed(42)

g = 9.81          # Gravity (m/s²)
velocity = 20     # Initial velocity (m/s)
angle = 45        # Launch angle (degrees)

theta = np.radians(angle)

# Horizontal distance
x = np.linspace(0, 40, 200)

# Theoretical projectile equation
y_true = (
    x * np.tan(theta)
    - (g * x ** 2)
    / (2 * velocity ** 2 * np.cos(theta) ** 2)
)

# Keep only values above ground
mask = y_true >= 0

x = x[mask]
y_true = y_true[mask]

# Add experimental noise
noise = np.random.normal(0, 0.5, len(y_true))

y = y_true + noise

X = x.reshape(-1, 1)

# -------------------------------------------------
# Build Neural Network
# -------------------------------------------------

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(1,)),

    tf.keras.layers.Dense(
        50,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        50,
        activation="relu"
    ),

    tf.keras.layers.Dense(1)

])

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

# -------------------------------------------------
# Train Model
# -------------------------------------------------

print("Training Neural Network...")

history = model.fit(
    X,
    y,
    epochs=2000,
    verbose=0
)

print("Training Completed!")

# -------------------------------------------------
# Prediction
# -------------------------------------------------

distance = float(
    input("Enter Horizontal Distance (m): ")
)

prediction = model.predict(
    np.array([[distance]]),
    verbose=0
)

print("\nPrediction")
print("-----------------------------")
print(f"Distance : {distance:.2f} m")
print(f"Predicted Height : {prediction[0][0]:.2f} m")

# -------------------------------------------------
# Evaluate Model
# -------------------------------------------------

predicted_height = model.predict(
    X,
    verbose=0
)

mse = np.mean(
    (y - predicted_height.flatten()) ** 2
)

print(f"\nMean Squared Error : {mse:.4f}")

# -------------------------------------------------
# Plot Results
# -------------------------------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    x,
    y,
    label="Experimental Data"
)

plt.plot(
    x,
    y_true,
    linewidth=3,
    label="Theoretical Curve"
)

plt.plot(
    x,
    predicted_height,
    linewidth=3,
    label="Machine Learning Prediction"
)

plt.title("Projectile Motion using Machine Learning")

plt.xlabel("Horizontal Distance (m)")
plt.ylabel("Height (m)")

plt.grid(True)

plt.legend()

plt.show()
