from PIL import Image
import tensorflow as tf

newmodel = tf.keras.models.load_model('modelo.h5')
print(newmodel.summary())


img = Image.open('zapato.jpg')
print(newmodel.predict(img))
