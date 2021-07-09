# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 13:32:04 2019
@author: jacqueline.cortez
"""
#Import packages
import pandas as pd                                                                 #For loading tabular data
from keras.layers import Dense                                                      #For DeapLearning
from keras.models import Sequential                                                 #For DeapLearning
from keras.utils import to_categorical                                              #For DeapLearning

#Reading data from files
file = 'mnist.csv'
mnist_df = pd.read_csv(file, header=None)
mnist_predictors = mnist_df.drop([0], axis=1).values
mnist_target = to_categorical(mnist_df[0]) # Convert the target to categorical: target
mnist_n_cols = mnist_predictors.shape[1] # Save the number of columns in predictors: n_cols
mnist_input_shape = (mnist_n_cols,)

#Create the model for prediction
mnist_model = Sequential() # Create the model: model
mnist_model.add(Dense(50, activation='relu', input_shape=mnist_input_shape)) # Add the first hidden layer
mnist_model.add(Dense(50, activation='relu')) # Add the second hidden layer
mnist_model.add(Dense(10, activation='softmax')) # Add the output layer
mnist_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Compile the model
mnist_model.fit(mnist_predictors, mnist_target, validation_split=0.3, epochs=10, batch_size=32) # Fit the model
print("Loss function: ", mnist_model.loss) # Verify that model contains information from compiling