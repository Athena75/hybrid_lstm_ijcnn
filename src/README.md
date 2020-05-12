
## Source Code:

 * **main.py**  : main program
It executes the following tasks:
1. loads dataset located in /data/raw
2. split the dataset into train and test subsets
3. Push each subset through preprocess pipeline to get latent representations from different pretrained encoders ("dense" and "lstm")
4. Make tsne visualizations of both the original dataset and the different latent representations: the plots are saved in **/results/visualizations/**
5. train different classifiers in the original dataset and transformed dataset
save performances results un a csv file containing Mean Square errors for each classifier and for each data representation (original, dense-encoded, lstm-encoded)
6. save the csv result in **/results/** and diplay it

 * Collecting: 
fetch and save locally datasets from a given url
datasets_names and urls of remote servers that hosts tha data 
(now we have only credit card fraud dataset)
the dataset is saved in /data/raw

 * Processing: 
contains a processing **Pipeline** which is a succession of processing modules/functions to tranfsorm data input to be adapted to our models. the pipeline add flexibility to the processing task so that it wouldn't be overwhelming to execute all process functions each time the dataset is updated

 * Modeling:  Scripts to train and evaluate models and then use trained models to make predictions/ classifications. it contains:
    * **neural_network_model.py** : module containing a metaclass composed of genral parameters and methods for training and saving neural network models 
    * **autoencoder.py** : subclass of neural network model : the autoencoder can be either 'dense' (classic neural network) or 'lstm' (recurrent nural network with lstm cells) 
 * Visualization : 
Scripts to create comprehensive visualizations. Here we have **visualizers.py** which contains a methods to gets tsne visualizations
