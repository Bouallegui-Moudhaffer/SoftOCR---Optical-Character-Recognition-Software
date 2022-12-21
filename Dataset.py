import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
import numpy as np, cv2
import os, glob
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
CATEGORIES = os.listdir('dataset')
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    zoom_range=(0.95, 0.95),
    horizontal_flip=False,
    vertical_flip=False,
    data_format='channels_last',
    validation_split=0.1,
    dtype=tf.float32,
)

train_generator = datagen.flow_from_directory(
    'dataset/',
    target_size=(96, 96),
    batch_size=16,
    color_mode='grayscale',
    class_mode='sparse',
    shuffle=True,
    subset='training',
    seed=123,
)
validation_generator = datagen.flow_from_directory(
    'test/',
    target_size=(96, 96),
    batch_size=16,
    color_mode='grayscale',
    class_mode='sparse',
    shuffle=True,
    subset='validation',
    seed=123,
)

model = Sequential()
# First Convolution Layer
model.add(
    Conv2D(64, (3, 3), input_shape=(96, 96, 1)))  # only for first convolution layer we mention layer size
model.add(Activation("relu"))  # activation function
model.add(MaxPooling2D(pool_size=(2, 2)))  # Maxpooling
# 2nd Convolution Layer
model.add(Conv2D(64, (3, 3)))  # only for first convolution layer we mention layer size
model.add(Activation("relu"))  # activation function
model.add(MaxPooling2D(pool_size=(2, 2)))  # Maxpooling
# 3rd Convolution Layer
model.add(Conv2D(64, (3, 3)))  # only for first convolution layer we mention layer size
model.add(Activation("relu"))  # activation function
model.add(MaxPooling2D(pool_size=(2, 2)))  # Maxpooling
# Fully Connected Layer # 1
model.add(Flatten())  # only for first convolution layer we mention layer size
model.add(Dense(64))  # activation function
model.add(Activation("relu"))  # Maxpooling
# Fully Connected Layer # 2
model.add(Dense(32))
model.add(Activation("relu"))
# Last Fully Connected Layer, output must be equal to number of classes, 10 (0-9)
model.add(Dense(81))  # this last dense Layer must be equal to 10
model.add(Activation('softmax'))  # activation function is changed to softmax (Class probabilities)

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(train_generator, epochs=30)  # Training my model
test_loss, test_acc = model.evaluate(validation_generator)
print("Test Loss on test samples", test_loss)
print("Validation Accuracy on test samples", test_acc)
predict_list = []

for img in glob.glob(rf"marks\*.png"):
    test_image = cv2.resize(cv2.imread(img, cv2.IMREAD_GRAYSCALE),  (96, 96))

    test_image = np.array(test_image).reshape(-1, 96, 96, 1)

    # predictions = np.argmax(model.predict(test_image), axis=-1)
    predictions = model.predict(test_image)
    pred_name = CATEGORIES[np.argmax(predictions)]
    predict_list.append([pred_name, img])
print(predict_list)

# Evaluating on testing dataset MNIST
# test_loss, test_acc = model.evaluate(train_generator)
# print("Test Loss on 10,000 test samples", test_loss)
# print("Validation Accuracy on 10,000 test samples", test_acc)

