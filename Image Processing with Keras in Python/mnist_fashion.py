# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 21:11:05 2019

@author: jacqueline.cortez
"""
# Import packages                          
import matplotlib.pyplot as plt                                               #For creating charts
import tensorflow as tf                                                       #For DeapLearning

from keras.layers                    import Conv2D                            #For DeapLearning
from keras.layers                    import Dense                             #For DeapLearning
from keras.layers                    import Dropout                           #For DeapLearning
from keras.layers                    import Flatten                           #For DeapLearning
from keras.layers                    import MaxPooling2D                      #For DeapLearning
from keras.models                    import Sequential                        #For DeapLearning
from keras.utils                     import plot_model                        #For DeapLearning
from keras.utils                     import to_categorical                    #For DeapLearning

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 #Data normalization

plt.pcolor(x_train[10],  cmap='gray') # Visualize the result
plt.gca().invert_yaxis()
plt.title('Figure 1')
plt.suptitle("0. Fashion MNIST Database")
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

plt.figure(figsize=(6,6))
for i in range(25):
 plt.subplot(5,5,i+1)
 plt.xticks([])
 plt.yticks([])
 plt.grid(False)
 plt.imshow(x_train[i], cmap=plt.cm.binary)
 plt.xlabel(class_names[y_train[i]])
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
plt.suptitle("0. Fashion MNIST Database")
plt.show()

model = Sequential() # Initialize the model object
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1), name='Input')) # Add a convolutional layer
model.add(MaxPooling2D(pool_size=2, name='MaxPooling2D'))
model.add(Dropout(0.3, name='Dropout'))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu', name='Conv2D'))
model.add(MaxPooling2D(pool_size=2, name='MaxPooling-2D'))
model.add(Dropout(0.3, name='Dropout-2'))
model.add(Flatten(name='Flatten')) # Flatten the output of the convolutional layer
model.add(Dense(256, activation='relu', name='Dense'))
model.add(Dropout(0.5, name='Dropout-3'))
model.add(Dense(10, activation='softmax', name='Output')) # Add an output layer for the 3 categories

# Take a look at the model summary
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train   = x_train.reshape(x_train.shape[0], 28, 28, 1)
y_train = to_categorical(y_train, len(class_names))

x_test   = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_test = to_categorical(y_test, len(class_names))

model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=.20)

# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])

plot_model(model, to_file='01_08_model.png', show_shapes=False, show_layer_names=True, rankdir='TB') # rankdir='TB' makes vertical plot and rankdir='LR' creates a horizontal plot

# Plotting the model
plt.figure()
data = plt.imread('01_08_model.png') # Display the image
plt.imshow(data)
plt.title('A MNIST case in a Neural Red')
plt.show()
