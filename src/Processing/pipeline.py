from sklearn import preprocessing 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
SRC_PATH = '../src'
MODELS_PATH = '../results/keras_models'

import sys
import os
import pickle
sys.path.insert(0, os.path.join(SRC_PATH,'Collecting'))
#sys.path.insert(0, os.path.join(SRC_PATH,'Processing'))
sys.path.insert(0, os.path.join(SRC_PATH,'Modeling'))
#from ..Modeling.helpers import *
#from ..Collecting import get_data
import numpy as np
from ..Modeling.autoencoder import AutoEncoderHelpers


class X_scaler_transformer(BaseEstimator, TransformerMixin):
    """Sklearn Transformer that scale up inputs using
    MinMaxScaler() of sklearn.preprocessing library 
    
    for more details:
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    """
    def __init__(self, seraparate_class=False):
        #self.scaling_method=='MinMaxScaler'
        self.seraparate_class=seraparate_class
        
    def fit(self, X, y=None):
        return self #nothing to estimate
        
    def transform(self, X, y=None):
        #apply  MinMaxScaler to X
        
        x_scaled = preprocessing.MinMaxScaler().fit_transform(X)
        if y and self.seraparate_class:
            np.c_[x_scaled[y == 0], x_scaled[y == 1]]
            
        return x_scaled

class latent_representation(BaseEstimator, TransformerMixin):
    """Sklearn estimator that fits (X,y) input to compute the corresponding encodes
    Attributes:
    - type:the type of used encoder have to be 'lstm' or 'dense'
    - random_seed: used by the autoencoder methods
    Methods : 
     fit(X, y) : get
     
     estimated variables :
         self.normal_idx, self.abnormal_idx : indexes of selected normal and abnormal entries to be encoded
         self.normal_hid_rep_, self.abnormal_hid_rep_ : the corresponding encoding
    """
    def __init__(self, type='lstm', random_seed=75): 
        self.type = type
        self.random_seed = random_seed
        
    def fit(self, X, y):
        model = AutoEncoderHelpers.restore(model_type=self.type+'_autoencoder')
        self.normal_idx, self.normal_hid_rep_ = model.get_latent_representation(X[y==0], random_seed=self.random_seed)
        self.abnormal_idx, self.abnormal_hid_rep_ = model.get_latent_representation(X[y==1], random_seed=self.random_seed)
        return self  
    
    def transform(self, X, y=None):
        return self

def _data_pipeline(data, target_column = 'Class', model_type='lstm', random_seed=75):
    #descriptive features
    X = data.drop([target_column], axis=1)
    #target values
    y = data[target_column].values
    #scale up descriptive features
    scalor = X_scaler_transformer()
    x_scaled = scalor.transform(X.values)
    #create a latent_representation that we will use to get 
    latent_representation_estimator = latent_representation(model_type, random_seed=random_seed)
    #fit the estimator
    latent_representation_estimator.fit(x_scaled, y)
    return latent_representation_estimator

def get_transformed_datasets(data, target_column = 'Class', random_seed=75):
    """Execute all the transformation steps in the right order for a given dataset input, 
    to get the different encodings from diffrent types of pretrained autoencoders (dense and lstm)
    
    Parameters:
    -----------
    data : train or test dataframe to go through the pipeline 
    target_column : name of the target feature
    random_seed : for random generator
    
    Returns:
    ---------
    a dictionary of (X,y) datasets couples:
        where X : descriptive features
              y : target variable
    
    the returned result is smth like this:          
    { 
        'original' : (x_original, y_original), 
        'dense' : (dense_rep_x, dense_rep_y),
        'lstm' : (lstm_rep_x, lstm_rep_y)
      }
    """
    datasets={}
    #create estimator to get encoding
    latent_representation_estimator = _data_pipeline(data, model_type='dense', random_seed=random_seed)
    x=data.drop(target_column, axis=1).values
    x_original = np.append( x[latent_representation_estimator.normal_idx,:],
                              x[latent_representation_estimator.abnormal_idx,:], 
                              axis = 0)
    y_normal = np.zeros(len(latent_representation_estimator.normal_idx))
    y_abnormal = np.ones(len(latent_representation_estimator.abnormal_idx))
    y_original = np.append(y_normal, y_abnormal)
    #add original subset
    datasets['original']=(x_original, y_original)

    #get encoding obtained with dense encoder
    dense_rep_x = np.append(latent_representation_estimator.normal_hid_rep_, 
                              latent_representation_estimator.abnormal_hid_rep_, 
                              axis = 0)
    y_normal = np.zeros(latent_representation_estimator.normal_hid_rep_.shape[0])
    y_abnormal = np.ones(latent_representation_estimator.abnormal_hid_rep_.shape[0])
    dense_rep_y = np.append(y_normal, y_abnormal)
    #add dense encodings
    datasets['dense']=(dense_rep_x, dense_rep_y)
    
    #get encoding obtained with lstm encoder
    latent_representation_estimator = _data_pipeline(data, model_type='lstm', random_seed=random_seed)
    lstm_rep_x = np.append( latent_representation_estimator.normal_hid_rep_, 
                              latent_representation_estimator.abnormal_hid_rep_, 
                              axis = 0)
    y_normal = np.zeros(latent_representation_estimator.normal_hid_rep_.shape[0])
    y_abnormal = np.ones(latent_representation_estimator.abnormal_hid_rep_.shape[0])
    lstm_rep_y = np.append(y_normal, y_abnormal)
    #add lstm encodings
    datasets['lstm']=(lstm_rep_x, lstm_rep_y)
    
    return datasets








