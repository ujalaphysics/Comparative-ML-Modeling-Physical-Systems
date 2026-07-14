"""
Project:
Comparative Machine Learning Modeling of Linear and Nonlinear Physical Systems

Linear Physical System:
Hooke's Law using TensorFlow

Author: Ujala
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# -------------------------------------
# Generate Dataset
# -------------------------------------

np.random.seed(42)

k = 5

extension = np.linspace(0, 10, 100)

noise = np.random.normal(0, 1, len(extension))

force = k * extension + noise

X = extension.reshape(-1, 1)
Y = force

# -------------------------------------
# Build Machine Learning Model
# -------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer="sgd",
    loss="mean_squared_error"
)

# -------------------------------------
# Train Model
# -------------------------------------

print("Training Model...")

history = model.fit(
    X,
    Y,
    epochs=500,
    verbose=0
)

print("Training Complete!")

# -------------------------------------
# Prediction
# -------------------------------------

new_extension = float(
    input("Enter Extension (m): ")
)

prediction = model.predict(
    np.array([[new_extension]]),
    verbose=0
)

print("\nPrediction Result")
print("----------------------------")
print(f"Extension : {new_extension:.2f} m")
print(f"Predicted Force : {prediction[0][0]:.2f} N")

# -------------------------------------
# Learned Parameters
# -------------------------------------

weights, bias = model.layers[0].get_weights()

print("\nLearned Equation")
print("----------------------------")

print(
    f"Force = {weights[0][0]:.3f} × Extension + {bias[0]:.3f}"
)

# -------------------------------------
# Evaluate Model
# -------------------------------------

predicted_force = model.predict(
    X,
    verbose=0
)

mse = np.mean(
    (Y - predicted_force.flatten()) ** 2
)

print(f"\nMean Squared Error : {mse:.4f}")

# -------------------------------------
# Plot
# -------------------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    extension,
    force,
    color="blue",
    label="Experimental Data"
)

plt.plot(
    extension,
    predicted_force,
    color="red",
    linewidth=3,
    label="ML Prediction"
)

plt.title("Hooke's Law using Machine Learning")

plt.xlabel("Extension (m)")
plt.ylabel("Force (N)")

plt.grid(True)

plt.legend()

plt.show()
