import tensorflow as tf
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
# red neuronal que nos ayudara a calcular el peso de una persona
# en base a su altura
# 2 entradas (altura, edad)
# 1 capa oculta de 3 neuronas
# 1 salida (peso)

# definimos las entradas
altura = np.array([1.60, 1.65, 1.70, 1.75, 1.80, 1.85,
                  1.90, 1.95, 2.00, 2.05], dtype=np.float32)
peso = np.array([60, 70, 80, 90, 100, 110, 120,
                130, 140, 150], dtype=np.float32)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(optimizer=tf.keras.optimizers.Adam(
    0.1), loss='mean_squared_error')

print("Comenzando entrenamiento...")
historial = modelo.fit(x=altura, y=peso, epochs=500, verbose=False)
print("Modelo entrenado")

print("Entrenamiento corriendo en el background")
print(modelo.predict([1.75, 1.80, 1.85, 1.90, 1.95, 2.00, 2.05]))
