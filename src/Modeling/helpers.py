# -*- coding: utf-8 -*-
""" This module contains some functions that wouls be useful to 
handle someMachine learning models utilities 
such as: revovering the last saved trained model ...
"""

import os
SRC_PATH = '../src'
MODEL_PATH = '../results/sklearn_models'
MODEL_EXTENSION = "h5"
from sklearn.metrics import mean_squared_error
#sys.path.insert(0, os.path.join(SRC_PATH,'Modeling'))
#from neural_network_model import NeuralNetwork
#from autoencoder import Autoencoder   

import pickle
import numpy as np
#from keras.models import load_model

class ClassifierHelpers(object):
    @classmethod
    def save(cls, clf, name, save_path=MODEL_PATH, model_extension='.sav'):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        filename = name+model_extension
        pickle.dump(clf, open(os.path.join(save_path, filename), 'wb'))
        
    @classmethod    
    def load(cls, filename, save_path=MODEL_PATH, model_extension='.sav'):
        binary_model = os.path.join(save_path, filename+model_extension)
        #return None if the model doesn't exist
        if not os.path.isfile(binary_model):
            return
        with open(binary_model, 'rb') as f:
            return pickle.load(f)
        
    @classmethod
    def evaluate(cls, clf, X_test, y_test):
        y_pred = clf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return np.sqrt(mse)
'''
def helpers_decorator(f):
        def wrapper(*args, **kwargs):
            assert (os.path.isdir(MODEL_PATH)), ">_< No saved binary models !! :(("
            return f(*args, **kwargs)
        return wrapper
    
class AutoEncoderHelpers(object):
        
    @helpers_decorator   
    def last_model_name(path = MODEL_PATH, model_format=MODEL_EXTENSION):
        """ Retrunrs last created binary model saved in <path>
        
        -----------
        Parameters
        -----------
        - path = where models are saved
            *default : MODEL_PATH
        - model_format = format of serialization 
                * default .h5 keras
       ------------
       Returns
       ----------
         name of the last saved binary model
        """
        binary_models = [model for model in os.listdir(path) if model.endswith(model_format)]
        sorted_by_date=sorted(binary_models, key=lambda name: os.path.getctime(os.path.join(path,name)))
        #if sorted_by_date is not empty
        if len(sorted_by_date)>0:
            #return last created model
            return sorted_by_date[-1]
        #else return None
        return
        
    @helpers_decorator
    def restore(model_name=None, model_format=MODEL_EXTENSION, model_type='dense_autoencoder'):
        """Restore the neural network model
        
        -----------
        Parameters
        -----------
        - model_name = name of the model to restore
            *if it's None : restore the last saved one
        - model_format = format of serialization 
                * default .h5 keras
        -model_type = the type of the model to be restored
            take values in ['dense_autoencoder', 'lstm_autoencoder']
       
       Returns:
       -None : if no model have been saved
       Else:
        (a pretrained) NeuralNetwork object  
        """
        #binary model is in '../../results/keras_models/<model_type>'
        models_dir = MODEL_PATH
        for dir in model_type.split('_')[::-1]:
            models_dir = os.path.join(models_dir, dir)
            
        if (model_name) and (model_name in os.listdir(models_dir)):
            #add h5 extension
            model_name += '.' + model_format
        #else get last saved model    
        else:
            model_name = AutoEncoderHelpers.last_model_name(path = models_dir, model_format=model_format)
            if not model_name:
                print(models_dir, "is empty :((( ")
                return
            
        #get model attributes
        with open(os.path.join(models_dir, model_name.split('.')[0]), 'rb') as f:
            my_depickler = pickle.Unpickler(f)
            attributes = my_depickler.load()
            
        #create model from restored attributes    
        if 'autoencoder' in model_type:     
            restored_model = Autoencoder(n_features=attributes['n_features'], nb_epoch = attributes['nb_epoch'], batch_size = attributes['batch_size'], 
                     validation_split = attributes['validation_split'], optimizer = attributes['optimizer'], loss = attributes['loss'],
                     autoencoder_type=attributes['autoencoder_type'])
        else:
            restored_model = NeuralNetwork(attributes)
        # load model weights    
        restored_model.model = load_model(os.path.join(models_dir, model_name))
        #print(model_name)
        return restored_model
'''

