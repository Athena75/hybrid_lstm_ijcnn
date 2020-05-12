import os
import pandas as pd
import zipfile 
import requests

SRC_PATH = '../src'
cwd=os.path.join(SRC_PATH,'Collecting')
#sys.path.insert(0, os.path.join(SRC_PATH,'Collecting'))
RAW_DATA_PATH = '../data/raw/'
DATASETS_INDEX =os.path.join(cwd,'datasets.csv')
DEFAULT_DATASET = "credit_card_fraud"

class collector_decorator(object):

    def __init__(self, func_to_decorate):
        self.func_to_decorate = func_to_decorate

    def __call__(self, *args, **kwargs):
        name = kwargs.get('dataset_name', DEFAULT_DATASET)
        print(':: Dataset Name ::', name)
        print("...looking for '{}' in referentiel data...".format(name))
        with open(os.path.abspath(DATASETS_INDEX), 'r') as f:
            names = [row.split(',')[0] for row in f]
        try:
            assert name in names
        except AssertionError:
            print(":( non existant dataset !! ")

        dataset_dir = os.path.join(RAW_DATA_PATH, name)
        #non existant repertory : return empty dataframe
        if not os.path.isdir(dataset_dir):
            msg = "Create {} directory".format(dataset_dir)
            print("*"*(len(msg)+6))
            print("   ", msg)
            print("*"*(len(msg)+6))
            os.makedirs(dataset_dir)
        return self.func_to_decorate(*args, **kwargs)
        
             
@collector_decorator
def fetch_data(dataset_name=DEFAULT_DATASET):
    """Donwloads the given dataset and save it in ./data/raw/
    - if the dataset already exists 
    """
          
    #where to deposit the downloaded file
    target_dir = os.path.join(RAW_DATA_PATH, dataset_name)
    #if the dataset already exists exit
    if len(os.listdir(path=target_dir))>0:
        print("datasets already downloded :) ")
        return
      
    #get dataset informations from ./datasets.csv
    resources = pd.read_csv(DATASETS_INDEX).set_index('name')
        
    url = resources.loc[dataset_name, 'url']
    
    if url.endswith('zip'):
        zip_file = os.path.join(target_dir, dataset_name+".zip")
        #download (large) zip file
        #for large https request on stream mode to avoid out of memory issues
        #see : http://masnun.com/2016/09/18/python-using-the-requests-module-to-download-large-files-efficiently.html
        print("**************************")
        print("  Downloading zip file")
        print("  >_<  Please wait >_< ")
        print("**************************")
        response = requests.get(url, stream=True)
        #read chunk by chunk
        handle = open(zip_file, "wb")
        for chunk in response.iter_content(chunk_size=512):
            if chunk:  
                handle.write(chunk)
        handle.close()  
        print("  Download completed ;) :") 
        #extract zip_file
        zf = zipfile.ZipFile(zip_file)
        print("1. Extracting {} file".format(dataset_name+".zip"))
        zf.extractall(path=target_dir)
        print("2. Deleting {} file".format(dataset_name+".zip"))
        os.remove(path=zip_file)
        print("=> Extracted data are located in \n     {}".format(target_dir))
        
    elif url.endswith('csv'):
        csv_file = os.path.join(target_dir, dataset_name+".csv")
        
        with open(csv_file,'w') as f:
            #get variables names
            print('Downloading csv file')
            lines = requests.get(url).content.decode("utf-8").split('\n')
            columns = [line for line in lines if '#' in line]
            columns = [var.split('.')[1].strip() for var in columns]
            #first write header in csv file
            f.write(','.join(columns)+'\n')
            #write retireved samples
            f.write('\n'.join(lines[len(columns):]))
        print("{} Downloaded successfully")
    else:
        #ToDo : implement fetch method for multiple tables datasets 
        return
      
@collector_decorator
def load_data(dataset_name=DEFAULT_DATASET):
    """loads csv datasest as a pandas dataframe 
    Returns 
    - Pandas.DataFrame if the data exists (has already been downloaded)
    - None if data don't exist
    """
    dataset_dir = os.path.join(RAW_DATA_PATH, dataset_name)
    #non existant repertory : return empty dataframe
    if not len(os.listdir(dataset_dir))>0:
        print("non existant dataset :( ")
        return 
    
    for filename in os.listdir(dataset_dir):
        if ".csv" in filename:
            return pd.read_csv(os.path.join(dataset_dir, filename))
    # no csv file return empty dataframe
    return pd.DataFrame()
    
