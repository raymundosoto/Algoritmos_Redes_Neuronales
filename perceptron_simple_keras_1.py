from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
import numpy as np

# 1. Crear datos de ejemplo (compuerta AND)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

#Este código crea un perceptrón simple para aprender la compuerta AND usando odulo Perceptron de sklearn.

# 2. Crear y entrenar el perceptrón
clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
clf.fit(X, y)

# 3. Evaluar
print("Precisión:", clf.score(X, y))

# 4. Ver pesos y bias
print("\nParámetros aprendidos:")
print("Pesos (W):", clf.coef_[0])       
print("Bias (b):", clf.intercept_[0])

# 4. Predecir
print("Predicciones:")
for i in range(len(X)):
    print(f"Entrada: {X[i]} -> Predicción: {clf.predict([X[i]])[0]}")
    
    