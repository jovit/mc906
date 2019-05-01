from read_data import read_data
from skimage import io
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import os

NUMBER_OF_FEATURES = 500

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

tf.random.set_random_seed(1)

mnist = tf.keras.datasets.mnist

train = read_data('imgs/train/').astype(float)
train = train / 255.

test = read_data('imgs/test/').astype(float)
test = test / 255.

train = np.array([it.flatten() for it in train])
test = np.array([it.flatten() for it in test])


# io.imshow((x_train[0] * 255).reshape((28,28)).astype(int), cmap='gray')
# plt.show()


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(50*50),

    tf.keras.layers.Dense(2000, activation=tf.nn.relu),

    tf.keras.layers.Dense(1000, activation=tf.nn.relu),

    tf.keras.layers.Dense(NUMBER_OF_FEATURES,  activation=tf.nn.sigmoid),

    tf.keras.layers.Dense(1000, activation=tf.nn.relu),

    tf.keras.layers.Dense(2000, activation=tf.nn.relu),

    tf.keras.layers.Dense(50*50, activation=tf.nn.sigmoid),

])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train, train, epochs=10, batch_size=10, validation_data=(test,test))

model.evaluate(test, test)


idx = 4  # index of desired layer
# a new input tensor to be able to feed the desired layer
layer_input = tf.keras.Input(shape=(NUMBER_OF_FEATURES,))

# create the new nodes for each layer in the path
x = layer_input
for layer in model.layers[idx:]:
    x = layer(x)

# create the model
new_model = tf.keras.Model(layer_input, x)
output = new_model.predict(np.full((1, NUMBER_OF_FEATURES), 0.5))
io.imshow((output[0] * 255.).reshape((50, 50)).astype(int), cmap='gray')
plt.show()

new_model.save('model1.h5')

print(output)
