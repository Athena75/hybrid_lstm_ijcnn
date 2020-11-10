## Overall project structure

Here is the explantation of folder strucure:
- **src**: Stores source code (python) 
- **results**: Folder for storing binary models, visualizations and models performances result
- **data**: Folder for storing subset data for experiments. It includes both raw data and processed data for temporary use.
- **notebook**: Storing all notebooks includeing EDA and modeling stage.

### src : Source Code:
* main.py : main program It executes the following tasks:
loads dataset located in /data/raw
split the dataset into train and test subsets
Push each subset through preprocess pipeline to get latent representations from different pretrained encoders ("dense" and "lstm")
Make tsne visualizations of both the original dataset and the different latent representations: the plots are saved in /results/visualizations/
train different classifiers in the original dataset and transformed dataset save performances results un a csv file containing Mean Square errors for each classifier and for each data representation (original, dense-encoded, lstm-encoded)
save the csv result in /results/ and diplay it
Collecting: fetch and save locally datasets from a given url datasets_names and urls of remote servers that hosts tha data (now we have only credit card fraud dataset) the dataset is saved in /data/raw

* Processing: contains a processing Pipeline which is a succession of processing modules/functions to tranfsorm data input to be adapted to our models. the pipeline add flexibility to the processing task so that it wouldn't be overwhelming to execute all process functions each time the dataset is updated

* Modeling: Scripts to train and evaluate models and then use trained models to make predictions/ classifications. it contains:

* neural_network_model.py : module containing a metaclass composed of genral parameters and methods for training and saving neural network models
autoencoder.py : subclass of neural network model : the autoencoder can be either 'dense' (classic neural network) or 'lstm' (recurrent nural network with lstm cells)
* Visualization : Scripts to create comprehensive visualizations. Here we have visualizers.py which contains a methods to gets tsne visualizations

To make it run:
```
#activate your venv (optional)
source ./venv/bin/activate

git clone https://github.com/Athena75/hybrid_lstm_ijcnn.git
pip install -r requirements.txt
cd src
python main.py
```

### results
It includes:

* Binary model + model metadata such as date, size of training data.
* Visualizations

### data:
It is composed of:
* raw: Storing the raw result which is generated from "preparation" folder code. My practice is storing a local subset copy rather than retrieving data from remote data store from time to time. It guarantees you have a static dataset for rest of action. Furthermore, we can isolate from data platform unstable issue and network latency issue.
* processed: To shorten model training time, it is a good idea to persist processed data. It should be generated from "processing" folder.
