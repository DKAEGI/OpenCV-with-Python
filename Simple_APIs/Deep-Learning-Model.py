### Deep Learning for Image Classification 
### Build an image classifier with Keras and Convolutional Neural Networks for the Fashion MNIST dataset. This data set includes 10 labels of different clothing types with 28 by 28 *grayscale* images. There is a training set of 60,000 images and 10,000 test images.
# 
#     Label	Description
#     0	    T-shirt/top
#     1	    Trouser
#     2	    Pullover
#     3	    Dress
#     4	    Coat
#     5	    Sandal
#     6	    Shirt
#     7	    Sneaker
#     8	    Bag
#     9	    Ankle boot
#     
#  

### The Data
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


### Visualizing the Data
import matplotlib.pyplot as plt

single_image = x_train[0]

plt.imshow(single_image)

# 9 for ankle boot
y_train[0]


### Preprocessing the Data
# Normalize the X train and X test data by dividing by the max value of the image arrays.

x_train.max()
x_test.max()

x_train = x_train/255 # divided by x_train_max
x_test = x_test/255


# Reshape the X arrays to include a 4 dimension of the single channel. Similar to what we did for the numbers MNIST data set.
x_train.shape
x_test.shape

x_train = x_train.reshape(60000, 28, 28, 1)
x_train.shape

x_test = x_test.reshape(10000,28,28,1)
x_test.shape


### Convert the y_train and y_test values to be one-hot encoded for categorical analysis by Keras.

from keras.utils import to_categorical
import numpy as np

# Number of classes in dataset
num_classes = len(np.unique(y_train))
num_classes

# One-hot encoded y_train and y_test
y_train_encoded = to_categorical(y_train, num_classes)
y_test_encoded = to_categorical(y_test, num_classes)


### Building the Model
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(28,28,1), activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))

# Last layer classifier, 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()


### Training the Model
# Train/Fit the model to the x_train set. Amount of epochs can be adapted.
model.fit(x_train,y_train_encoded,epochs=10)


### Evaluating the Model
model.metrics_names

model.evaluate(x_test,y_test_encoded)

from sklearn.metrics import classification_report

predictions = model.predict_classes(x_test)

y_test_encoded.shape

y_test_encoded[0]

predictions[0]

y_test

print(classification_report(y_test,predictions))

