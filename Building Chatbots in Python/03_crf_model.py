# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 13:55:46 2020

@author: jaces
"""
from sklearn_crfsuite import CRF #conditional random fields
import pickle #pickle.format_version

"""
crf = CRF(algorithm='lbfgs', 
          all_possible_states=None,
          all_possible_transitions=True, 
          averaging=None, 
          c=None, 
          c1=1, 
          c2=0.001,
          calibration_candidates=None, 
          calibration_eta=None,
          calibration_max_trials=None, 
          calibration_rate=None,
          calibration_samples=None, 
          delta=None, 
          epsilon=None, 
          error_sensitive=None,
          gamma=None, 
          keep_tempfiles=None, 
          linesearch=None, 
          max_iterations=50,
          max_linesearch=None, 
          min_freq=None, 
          model_filename=None,
          num_memories=None, 
          pa_type=None, 
          period=None, 
          trainer_cls=None,
          variance=None, 
          verbose=False)
"""
crf = CRF(algorithm='lbfgs', 
          all_possible_transitions=True, 
          c1=1, 
          c2=0.001,
          max_iterations=50,
          verbose=False)

filename = 'data/crf_model.pkl'
pickle.dump(crf, open(filename,'wb'))

with open(filename, 'rb') as f:  model = pickle.load(f)
print(model)
