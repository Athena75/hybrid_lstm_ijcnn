
## Collecting package

**1.  datasets.csv:** Public datasets metadata:
Attribute Information: 
* name: name of the dataset:
    - processing functions will create local directory (in ./data/raw) with the same name, where to store the dataset 
* url : server's url where the dataset is stored
* one_table : whether the dataset is composed of only one table or multiple
    * 1 : only one table
    * 0 : multiple
* labelled : whether the dataset is labelled or not (has a target vvariable or not)
* description : specify what the dataset is about and some additional informations..

**2.  get_data.py:** 
Module containing functions to fetch and load data
it handles also large datasets, and multi-tables datasets.

