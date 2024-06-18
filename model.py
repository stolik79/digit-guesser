import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.image import resize

(train_X, train_y), (test_X, test_y) = mnist.load_data()
def canvas_size(images, new_size=(60, 60)):
    images_resized = np.zeros((images.shape[0], new_size[0], new_size[1], 1))
    for i in range(images.shape[0]):
        img = tf.image.resize(images[i].reshape(28, 28, 1), new_size)
        images_resized[i] = img
    return images_resized

train_X = canvas_size(train_X)
test_X = canvas_size(test_X)
train_X = train_X.astype('float32') / 255.0
test_X = test_X.astype('float32') / 255.0
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(60, 60, 1)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=5)
model.save('model.h5')
