# -*- coding: utf-8 -*-
import pandas            as pd                                                      #For loading tabular data
import numpy             as np                                                      #For making operations in lists
import tensorflow        as tf
import matplotlib.pyplot as plt                                                     #For creating charts

from keras.callbacks  import EarlyStopping                                          #For DeapLearning
from keras.layers     import BatchNormalization                                     #For DeapLearning
from keras.layers     import Concatenate                                            #For DeapLearning
from keras.layers     import Dense                                                  #For DeapLearning
from keras.layers     import Embedding                                              #For DeapLearning
from keras.layers     import Flatten                                                #For DeapLearning
from keras.layers     import Input                                                  #For DeapLearning
from keras.models     import Model                                                  #For DeapLearning
from keras.optimizers import Adam                                                   #For DeapLearning
from keras.utils      import plot_model                                             #For DeapLearning

file = "games_season.csv"
games_season = pd.read_csv(file)

file = "games_tourney.csv"
games_tourney = pd.read_csv(file)

np.random.seed(1)
tf.set_random_seed(1)
pd.options.display.float_format = '{:,.4f}'.format 

team_in_1 = Input(shape=(1,), name='Team-1-In') # Create an Input for each team
team_in_2 = Input(shape=(1,), name='Team-2-In')

n_teams = np.unique(games_season['team_1']).shape[0] # Count the unique number of teams
team_lookup = Embedding(input_dim=n_teams, output_dim=1, input_length=1, name='Team-strength') # Create an embedding layer

team_1_strength = team_lookup(team_in_1) # Lookup the team inputs in the team strength model
team_2_strength = team_lookup(team_in_2)

normalize_1 = BatchNormalization(name='Normalization-1')(team_1_strength)
normalize_2 = BatchNormalization(name='Normalization-2')(team_2_strength)

flatten_1 = Flatten(name='Flatten-1')(normalize_1) # Flatten the output
flatten_2 = Flatten(name='Flatten-2')(normalize_2) # Flatten the output

home_in = Input(shape=(1,), name='Home-In') # Create an input for home vs away

out_concatenate = Concatenate(name='Gathering-together')([flatten_1, flatten_2, home_in]) # Combine the team strengths with the home input using a Concatenate layer, then add a Dense layer

output_tensor_1 = Dense(1, activation='linear', use_bias=False, name='Regression_out')(out_concatenate) # Create the first output
output_tensor_2 = Dense(1, activation='sigmoid', use_bias=False, name='Classification_out')(output_tensor_1) # Create the second output (use the first output as input here)

model = Model([team_in_1, team_in_2, home_in], [output_tensor_1, output_tensor_2]) # Make a Model
model.compile(optimizer=Adam(lr=0.01), loss='mean_absolute_error') # Compile the model

# Summarazing the model
print(model.summary()) # Summarize the model
plot_model(model, to_file='03_04_model.png', show_shapes=False, show_layer_names=True, rankdir='TB') # rankdir='TB' makes vertical plot and rankdir='LR' creates a horizontal plot

# Plotting the model
plt.figure()
data = plt.imread('03_04_model.png') # Display the image
plt.imshow(data)
plt.title('"Shared Layers Model" (Using "team_strength_model")')
plt.suptitle("9. Add the model predictions to the tournament data")
#plt.subplots_adjust(left=0.2, bottom=None, right=None, top=0.88, wspace=0.3, hspace=None)
plt.show()

model.fit([games_season['team_1'], games_season['team_2'], games_season['home']], 
          [games_season['score_diff'], games_season['won']],
          epochs=150, verbose=False, validation_split=.10, batch_size=2048, callbacks=[EarlyStopping(patience=2)])
predictions = np.array(model.predict([games_tourney['team_1'], games_tourney['team_2'], games_tourney['home']]))
games_tourney['sc_diff_pred'] = predictions[0,:,0]
games_tourney['won_pred'] = predictions[1,:,0]
print(games_tourney[['score_diff','sc_diff_pred','won','won_pred']].head())

print(model.get_weights()) # Print the model weights
print(games_tourney.mean()) # Print the training data means

# Evaluate the model on new data
print(model.evaluate([games_season['team_1'], games_season['team_2'], games_season['home']], 
                     [games_season['score_diff'], games_season['won']],  
                     verbose=False))
