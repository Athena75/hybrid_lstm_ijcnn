from keras.layers import Input, Dense, LSTM, RepeatVector
from keras.models import Model, Sequential
from keras import regularizers
from ..Modeling.neural_network_model import NeuralNetwork
import numpy as np 
np.random.seed(42)

SRC_PATH = '../src'
import os
import sys
sys.path.insert(0, os.path.join(SRC_PATH,'Processing'))
#import pipeline
#import preprocessors
import pickle
MODELS_PATH = '../results/keras_models'
MODEL_PATH = MODELS_PATH
MODEL_EXTENSION = "h5"
from keras.models import load_model


class Autoencoder(NeuralNetwork):
    """docstring for Autoencoder"""
    def __init__(self, n_features, nb_epoch = 20, batch_size = 256, 
                 validation_split = 0.01, optimizer = 'adadelta', loss = 'mse',
                 autoencoder_type='dense', model=None):
        '''construct an autoencoder (dense or sequence-to-sequence autoencoder )
        
        Parameters:
        - n_feature : input dim (number of features without target feature)
        - nb_epoch : number of epochs for gradient decent  
            One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.
        - batch_size : Total number of training examples present in a single batch.
        - validation_split : fraction of the validation dataset 
        - optimizer : optimizer for gradient descent 
        - loss : loss function to minimize during training
        - model_architecture : dictionary defining for each type of the autoencoder the architecture of the neural network 
            - dense : list of n_neurons for each layer the first half is for the encoder
            - lstm : list 2 elments : latent_dim, timesteps
                    the encoder is composed of <latent_dim> units
                
        '''
        NeuralNetwork.__init__(self, n_features, nb_epoch, batch_size ,validation_split, optimizer, loss, model)
        
        self.autoencoder_type=autoencoder_type
        self.model_architecture={'dense':[100,50,50,100], #[n_neurons_layer1, n_neurons_layer2, 
                                 'lstm': [150,10]}#latent_dim, timesteps
        #train with only 2000 sample
        #because with auto encoder only few samples are enough
        self.train_sample= 2000
                                 
    def create(self):
        """ Builds auto_encoder layers according to 'model_architecture'"""
        if self.autoencoder_type=='dense':
            ## input layer 
            input_layer = Input(shape=(self.n_features,))
            layers=self.model_architecture['dense']

            for i, n_neurons in enumerate(layers):
                ## encoding part
                if i==0:
                    encoded = Dense(n_neurons, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)
                elif i<len(layers)/2-1:
                    encoded = Dense(n_neurons, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(encoded)
                elif i==len(layers)/2-1:
                    encoded = Dense(n_neurons, activation='relu')(encoded)
                ## decoding part
                elif i>len(layers)/2-1 and i<len(layers)-1:
                    decoded = Dense(n_neurons, activation='tanh')(encoded)

                else: #i==len(layers)-1
                    decoded = Dense(n_neurons, activation='tanh')(decoded)
            ## output layer
            output_layer = Dense(self.n_features, activation='relu')(decoded)
            self.model = Model(input_layer, output_layer)
            
        else:
            latent_dim, timesteps = self.model_architecture['lstm']
            input_layer = Input(shape=(timesteps, self.n_features))
            encoded = LSTM(latent_dim)(input_layer)

            decoded = RepeatVector(timesteps)(encoded)
            decoded = LSTM(self.n_features, return_sequences=True)(decoded)

            self.model = Model(input_layer, decoded)
        
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def fit(self, x_normal_scaled, sample=2000):
        """ fits the model to the normalized training data <x_normal_scaled>
        - sample  : number of random samples to select from x_normal_scaled"""
        if not sample:
            sample = self.train_sample
        #random samples
        rand_idx=np.random.randint(x_normal_scaled.shape[0], size=sample)
        x_normal_samples = x_normal_scaled[rand_idx, :]
        
        if self.autoencoder_type=="lstm":
            timesteps= self.model_architecture['lstm'][1]
            x_normal_samples = x_normal_samples.reshape((sample//timesteps, timesteps, x_normal_scaled.shape[1]))
        
        
        # for an autoencoder, the target is the same as the input
        self.model.fit(x_normal_samples, x_normal_samples, 
                        batch_size = self.batch_size, epochs = self.nb_epoch, 
                        shuffle = True)#, validation_split = self.validation_split)
        
    def _get_latent_representation(self,input_samples):
        #get the representaions from the encoder
        
        hidden_representation = Sequential()
        for i in range((len(self.model_architecture[self.autoencoder_type])//2)+1):
            hidden_representation.add(self.model.layers[i])
            
        if self.autoencoder_type=='dense':
            return hidden_representation.predict(input_samples[:self.train_sample])
        
        #else it's lstm  
        timesteps = self.model_architecture['lstm'][1]
        rand_idx=np.random.randint(input_samples.shape[0], size=self.train_sample)
        reshaped_input = input_samples[rand_idx,:].reshape((self.train_sample//timesteps, timesteps, input_samples.shape[1]))
        return hidden_representation.predict(reshaped_input)
    
    def get_latent_representation(self,input_samples, random_seed=42):
        #get the representaions from the encoder
        np.random.seed(random_seed)
        if len(input_samples)>=self.train_sample:
            selected_samples = input_samples[:self.train_sample]
            selected_indexes = range(self.train_sample)
        else:
            selected_indexes=np.random.randint(input_samples.shape[0], size=self.train_sample)
            selected_samples = input_samples[selected_indexes,:]
        
        hidden_representation = Sequential()
        for i in range((len(self.model_architecture[self.autoencoder_type])//2)+1):
            hidden_representation.add(self.model.layers[i])
 
        if self.autoencoder_type=='dense':
            return selected_indexes, hidden_representation.predict(selected_samples)
        
        #else it's lstm  
        timesteps = self.model_architecture['lstm'][1]
        reshaped_input = selected_samples.reshape((self.train_sample//timesteps, timesteps, input_samples.shape[1]))
        return selected_indexes, hidden_representation.predict(reshaped_input)
    
    def save(self):
        save_path = os.path.join(MODELS_PATH, 'autoencoder')
        save_path = os.path.join(save_path,self.autoencoder_type)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        NeuralNetwork.save(self, name_prefix=self.autoencoder_type+'_autoencoder_', path = save_path)
        #self.save(self, name=None, path = MODELS_PATH)
 
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
       
    
        
