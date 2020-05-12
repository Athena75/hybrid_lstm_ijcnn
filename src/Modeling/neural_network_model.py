""" Define neural network metaclass including general methods and attributes """

SRC_PATH = '../src'
MODELS_PATH = '../results/keras_models'
MODEL_EXTENSION = "h5"
import sys
import os
import pickle
#sys.path.insert(0, os.path.join(SRC_PATH,'Processing'))

#sys.path.insert(0, os.path.join(SRC_PATH,'Modeling'))
#from helpers import AutoEncoderHelpers, helpers_decorator

def last_model_name(path = MODELS_PATH, model_format=MODEL_EXTENSION):
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

class NeuralNetwork(object):
    """docstring for NeuralNetwork"""
    def __init__(self, n_features, nb_epoch = 20, batch_size = 256, 
                 validation_split = 0.01, optimizer = 'adadelta', loss = 'mse', model=None):
        '''     
        ----------
        Parameters:
        ----------
        - n_feature : input dim (number of features without target feature)
        - nb_epoch : number of epochs for gradient decent  
            One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.
        - batch_size : Total number of training examples present in a single batch.
        - validation_split : fraction of the validation dataset 
        - optimizer : optimizer for gradient descent 
        - loss : loss function to minimize during training
        '''
        self.n_features=n_features
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.validation_split = validation_split #0.05
        self.optimizer = optimizer
        self.loss = loss
        self.model = model
        
    def save(self, name_prefix='neural_network_', path = MODELS_PATH, model_format='.h5'):
        """Save binary model in h5 format
        -----------
        Parameters
        -----------
        -path : where to save the binary
        (default path : models_path)
        
        -name : model file name
            if name=None it will be saved automatically <name_prefix>_i.h5 (last saved model index+1 ) 
        
        """
        #last_model = AutoEncoderHelpers.last_model_name(path = path)
        last_model = last_model_name(path = path)
        # if it is the first h5 model
        if not last_model:
            model_name = name_prefix+'1'
        else:
            last_model = last_model.split('.')[0]
            model_name = name_prefix+str(int(last_model.split('_')[-1])+1)
                
        with open(os.path.join(path, model_name), 'wb') as f:
            my_pickler = pickle.Pickler(f)
            my_pickler.dump(self.__dict__) 
            
        #add h5 extension    
        model_name = model_name + model_format
        print("==> Saving'{}' in {}".format(model_name, path)) 
        #save model in the path
        self.model.save(os.path.join(path, model_name))
        

