from sklearn import preprocessing 
from sklearn.model_selection import train_test_split

def elapsed_seconds_per_day(x):
    """This function is used only in credit_card_fraud """
    return x / 3600 % 24

def split_train_test(data, test_frac=0.2, random_seed=42, target_variable=None):
    """Splits original dataframe into random train and test subsets
    
    Parameters
    -----------
    - data: original dataframe to be splitted
    - test_frac :  float between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
    - random_seed : the seed used by the random number generator
    - target_variable : name of the class variable
        if it is given the subsesets will be be splitted into descriptive features and target feature
        
    Returns
    --------
    tuple of numpy arrays of train-test subsets split of data input
    ^ if target_variable is given returns separately X (explanatory variables) and y (target variable) for both train and test subsets
    
    """
    train_set, test_set = train_test_split(data, test_size=test_frac, random_state=random_seed)
    #reset indexes for both datasets
    train_set = train_set.reset_index(drop=True) 
    test_set = test_set.reset_index(drop=True)
    
    #separate X and y variables if target_variable is given
    if target_variable:
        X_train = train_set.drop(target_variable, axis=1)
        y_train = train_set[target_variable]
        X_test = test_set.drop(target_variable, axis=1)
        y_test = test_set[target_variable]
        return X_train, y_train, X_test, y_test
    
    # if target_variable is None return train and test subsets
    return train_set, test_set

def sample_train_data(data, normal_frac=1000, target_column='Class', random_seed=42):
    """Select random samples from the given dataset
    
    Parameters
    -----------
    - data : pandas.DataFrame : original dataset
    - normal_frac: int representing the number of "normal" labelled records to select
        ^in this context "normal" observation is labelled 0
    - target_colum : name of target variable
    - random_seed : the seed used by the random number generator
    
    Returns
    ----------
    (X, y) : X : numpy.array of selected samples without label
             y : numpy.array corresponding labels (0 <-> normal ; 1 <-> abnormal)"""
    
    normal = data[data[target_column] == 0].sample(normal_frac, random_state=random_seed)
    abnormal = data[data[target_column] == 1]
    #append normal and abnormal entries 
    #then get random samples
    df = normal.append(abnormal).sample(frac=1).reset_index(drop=True)
    
    #split the dataframe to descriptive features, and target feature
    X = df.drop([target_column], axis = 1).values
    y = df[target_column].values
    return X, y

def X_scaler(data, target_column='Class'):
    """ Transforms descriptuve features by scaling each feature to [0,1].
    
    Returns
    --------
    tuple con
    (x_normal_scaled,x_abnormal_scaled) 
    """
    #descriptive features
    X = data.drop([target_column], axis=1)
    #target values
    y = data[target_column].values
    #apply  MinMaxScaler to X
    #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    x_scaled = preprocessing.MinMaxScaler().fit_transform(X.values)
    #x_norm, x_fraud = x_scale[y == 0], x_scale[y == 1]
    return x_scaled[y == 0], x_scaled[y == 1]

