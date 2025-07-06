import numpy as np
import matplotlib.pyplot as plt

# Perceptrón simple para la compuerta lógica AND
# Cerrar figura al finalizar para terminar el prpgrama correctamente

X = np.array([           # 1. Datos de entrenamiento: Compuerta AND   
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 0, 0, 1])

# 2. Parámetros del perceptrón
tasa_aprendizaje = 0.1
max_epocas = 1000


num_entradas = X.shape[1]              # 3. Inicialización de pesos y bias (rango [-0.5, 0.5])
pesos = np.random.rand(num_entradas) - 0.5  # [-0.5, 0.5)
bias = np.random.rand() - 0.5              # [-0.5, 0.5)

print("Inicialización:")
print(f"  Pesos iniciales: {pesos}")
print(f"  Bias inicial: {bias:.4f}")


errores_por_epoca = []   # 4. Historial para gráficas

# 5. Entrenamiento del perceptrón
convergencia = False
for epoca in range(max_epocas):
    errores = 0
    
    
    for i in range(len(X)):             # Iterar sobre todos los patrones de entrenamiento
        # Calcular suma ponderada
        suma_ponderada = np.dot(X[i], pesos) + bias
        
        
        prediccion = 1 if suma_ponderada >= 0 else 0      # Aplicar función de activación (escalón)
        
        
        error = y[i] - prediccion   # Calcular error
        
        # Si hay error, actualizar pesos y bias
        if error != 0:
            errores += 1
            pesos += tasa_aprendizaje * error * X[i]
            bias += tasa_aprendizaje * error
    
    
    errores_por_epoca.append(errores) # Registrar errores de esta época
    
    # Verificar convergencia
    if errores == 0:
        print(f"\n¡Convergencia alcanzada en la época {epoca+1}!")
        convergencia = True
        break

# 6. Resultados finales
print("\nResultados después del entrenamiento:")
print(f"  Pesos finales: {pesos}")
print(f"  Bias final: {bias:.4f}")
print(f"  Total de épocas: {epoca+1}")

# 7. Probar el perceptrón entrenado
print("\nPruebas finales:")
for i in range(len(X)):
    suma_ponderada = np.dot(X[i], pesos) + bias
    prediccion = 1 if suma_ponderada >= 0 else 0
    print(f"  Entrada: {X[i]} -> Predicción: {prediccion} (Esperado: {y[i]})")

plt.figure(figsize=(10, 6))           # 8. Gráfica de evolución del error
plt.plot(errores_por_epoca, 'b-', linewidth=2)
plt.title('Evolución del Error durante el Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Número de Errores')
plt.grid(True)
plt.show()

|
if num_entradas == 2:                # 9. Gráfica de la recta de decisión (solo para 2D)
    plt.figure(figsize=(8, 6))
    
    # Graficar puntos de datos
    plt.scatter(X[y==0, 0], X[y==0, 1], c='red', s=100, label='Clase 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', s=100, marker='s', label='Clase 1')
    
    # Calcular recta de decisión (w1*x + w2*y + b = 0)
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5
    
    # Puntos para la línea de decisión
    if pesos[1] != 0:  # Evitar división por cero
        x_vals = np.array([x_min, x_max])
        y_vals = (-pesos[0] * x_vals - bias) / pesos[1]
        plt.plot(x_vals, y_vals, 'k--', linewidth=2)
    
    plt.title('Recta de Decisión del Perceptrón')
    plt.xlabel('Entrada X1')
    plt.ylabel('Entrada X2')
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.grid(True)
    plt.legend()
    plt.show()