import math
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
datos, metadatos = tfds.load(
    'fashion_mnist', as_supervised=True, with_info=True)

datos_entrenamiento, datos_prueba = datos['train'], datos['test']

nombres_clases = metadatos.features['label'].names


def normalizar(imagenes, etiquetas):
    imagenes = tf.cast(imagenes, tf.float32)
    imagenes /= 255
    return imagenes, etiquetas


datos_entrenamiento = datos_entrenamiento.map(normalizar)
datos_prueba = datos_prueba.map(normalizar)

datos_prueba = datos_prueba.cache()
datos_entrenamiento = datos_entrenamiento.cache()

modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
modelo.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])


num_ejemplos_entrenamiento = metadatos.splits['train'].num_examples
num_ejemplos_prueba = metadatos.splits['test'].num_examples
# print(num_ejemplos_entrenamiento)

batch_size = 32

datos_entrenamiento = datos_entrenamiento.repeat().shuffle(
    num_ejemplos_entrenamiento).batch(batch_size)
datos_prueba = datos_prueba.batch(batch_size)

historial = modelo.fit(datos_entrenamiento, epochs=5, steps_per_epoch=math.ceil(
    num_ejemplos_entrenamiento/batch_size))

for imaganes, etiquetas in datos_prueba.take(1):
    imaganes = imaganes.numpy()
    etiquetas = etiquetas.numpy()
    predicciones = modelo.predict(imaganes)
    print('predicciones: ', nombres_clases[np.argmax(predicciones[0])])
modelo.save('modelo.h5')
