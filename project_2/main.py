import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from skimage import io

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = np.array([it.flatten() for it in x_train])
x_test = np.array([it.flatten() for it in x_test])


# io.imshow((x_train[0] * 255).reshape((28,28)).astype(int), cmap='gray')
# plt.show()



model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(28*28),

  tf.keras.layers.Dense(250, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(50,  activation=tf.nn.sigmoid),

  tf.keras.layers.Dense(250, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(28*28, activation=tf.nn.sigmoid, name='out'),
])

model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, x_train, epochs=5)

model.evaluate(x_test, x_test)



idx = 4  # index of desired layer
layer_input = tf.keras.Input(shape=(50,)) # a new input tensor to be able to feed the desired layer

# create the new nodes for each layer in the path
x = layer_input
for layer in model.layers[idx:]:
    x = layer(x)

# create the model
new_model = tf.keras.Model(layer_input, x)
output = new_model.predict(np.full((1,50), 0.5))
io.imshow((output[0] * 255.).reshape((28,28)).astype(int), cmap='gray')
plt.show()

new_model.save('model1.h5')

print(output)
