
from sklearn.datasets import make_classification
import numpy as np
from sklearn.neural_network import MLPClassifier

#programa para crear un perceptrón simple para aprender la compuerta AND. con sklearn y modulo MLPClassifier
# 1. Crear datos de ejemplo (compuerta AND)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

    


# 1. Configurar un MLP con una sola neurona (equivalente a perceptrón)
mlp = MLPClassifier(hidden_layer_sizes=(),  # Sin capas ocultas
                    activation='logistic', # Función sigmoide
                    solver='sgd',          # Descenso de gradiente
                    learning_rate_init=0.1,
                    max_iter=1000,
                    random_state=42)

# 2. Entrenar
mlp.fit(X, y)

# 3. Evaluar
print("Precisión MLP:", mlp.score(X, y))

# 4. Ver pesos y bias
print("\nParámetros aprendidos:")
print("Pesos (W):", mlp.coefs_[0].flatten())
print("Bias (b):", mlp.intercepts_[0])

# 5. Predecir
print("Predicciones:")
for i in range(len(X)):
    print(f"Entrada: {X[i]} -> Predicción: {mlp.predict([X[i]])[0]}")