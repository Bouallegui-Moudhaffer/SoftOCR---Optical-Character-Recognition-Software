# Imports:
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
import numpy as np, cv2
import os, glob
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def cnn_model():
    # Adding data augmentation options
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=5,
        zoom_range=(0.95, 0.95),
        horizontal_flip=False,
        vertical_flip=False,
        data_format='channels_last',
        validation_split=0.1,
        dtype=tf.float32,
    )
    # Loading training set into a generator and defining its parameters; size, color_mode,
    # shuffle: feeds images of different classes everytime to make sure the model doesn't simply memorize the data
    train_generator = datagen.flow_from_directory(
        'Dataset/Training',
        target_size=(96, 96),
        batch_size=16,
        color_mode='grayscale',
        class_mode='sparse',
        shuffle=True,
        subset='training',
        seed=123,
    )
    # Loading validation set into a generator, this set will be responsible for testing the model
    # on new never before seen data to assess its true accuracy and check if it generalized well or not
    validation_generator = datagen.flow_from_directory(
        'Dataset/Validation',
        target_size=(96, 96),
        batch_size=16,
        color_mode='grayscale',
        class_mode='sparse',
        shuffle=True,
        subset='validation',
        seed=123,
    )
    # Adding Layers to the model
    model = Sequential()
    # First Convolution Layer
    # Multiplying by 64 filters
    model.add(
        Conv2D(64, (3, 3), input_shape=(96, 96, 1)))  # only for first convolution layer we mention layer size
    model.add(Activation("relu"))  # Best activation function
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Maxpooling
    # 2nd Convolution Layer
    model.add(Conv2D(64, (3, 3)))  # only for first convolution layer we mention layer size
    model.add(Activation("relu"))  # activation function
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Maxpooling
    # 3rd Convolution Layer
    model.add(Conv2D(64, (3, 3)))  # only for first convolution layer we mention layer size
    model.add(Activation("relu"))  # activation function
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Max-Pooling
    # Fully Connected Layer # 1
    model.add(Flatten())  # reduce matrix dimension
    model.add(Dense(128))  # Fully connected layer
    model.add(Activation("relu"))  # Activation
    # Fully Connected Layer # 2
    model.add(Dense(64))
    model.add(Activation("relu"))
    # Last Fully Connected Layer, output must be equal to number of classes, 84 (0-20)
    model.add(Dense(81))  # this last dense Layer must be equal to the number of classes
    model.add(Activation('softmax'))  # activation function is changed to softmax (Class probabilities)

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.fit(train_generator, epochs=20)  # Training my model
    # Evaluating on validation dataset
    test_loss, test_acc = model.evaluate(validation_generator)
    print("Test Loss on test samples", test_loss)
    print("Validation Accuracy on test samples", test_acc)
    # Saving Model
    # Serialize model to JSON
    model_json = model.to_json()
    with open("H5_FILE.json", "w") as json_file:
        json_file.write(model_json)
    # Serialize weights to HDF5
    model.save_weights("H5_FILE.h5")
    print("Saved model to disk")


def predict_digits(model, img):
    CATEGORIES = os.listdir('dataset')
    # Read image
    test_image = cv2.resize(cv2.imread(img, cv2.IMREAD_GRAYSCALE), (96, 96))
    # Reshaping the input image
    test_image = np.array(test_image).reshape(-1, 96, 96, 1)
    # Calling predict method on the image
    predictions = model.predict(test_image)
    # Getting the highest probability's label
    pred_name = CATEGORIES[np.argmax(predictions)]
    return pred_name
