SRC_PATH = '.'


from src.Visualization.visualizers import tsne_plot
from src.Modeling.autoencoder import *
from src.Modeling.helpers import *
from src.Collecting import get_data

from src.Processing import pipeline

from src.Processing import preprocessors
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.linear_model import SGDClassifier

import pickle

MODELS_PATH = '../results/sklearn_models'
RESULTS_PATH = '../results/'
VISUALIZATIONS_PATH = '../results/visualizations'
DATASET_NAME = "credit_card_fraud"
random_seed = 42
target_column = 'Class'

classifiers = {'sgd_clf': SGDClassifier(),
               'svc_clf': LinearSVC(C=1, loss="hinge", random_state=42),
               'svm_clf': svm.SVC(kernel='linear')}


def train_model(data, encode_type='dense'):
    print('train {} autoencoder'.format(encode_type))
    X, y = preprocessors.sample_train_data(data, random_seed=random_seed)
    scalor = pipeline.X_scaler_transformer(seraparate_class=True)
    x_normal, _ = scalor.transform(X, y)
    model = Autoencoder(n_features=X.shape[1], autoencoder_type=encode_type)
    model.create()
    model.fit(x_normal_scaled=x_normal)
    model.save()


if __name__ == '__main__':
    # get dataset
    print(" ::: Loading data :::")
    print(" >_< Please Wait >_< ")
    print()
    data = get_data.load_data(dataset_name=DATASET_NAME)
    print(" data loaded sucessfully ;)")
    # if '../../results/keras_models' is empty (no pretrained auto encoder) we have to train and save
    # train_model(data=data, encode_type='dense')
    # train_model(data=data, encode_type='lstm')

    print("1. train, test split\n")
    train, test = preprocessors.split_train_test(data, test_frac=0.2, random_seed=42)
    # push train and test to the pipeline
    print("2. transform train, test subsets to get latent representations\n")
    print(" 2.1- Compute train encodings \n")
    transformed_train = pipeline.get_transformed_datasets(train)
    print(" 2.2- Compute test encodings \n")
    transformed_test = pipeline.get_transformed_datasets(test)

    print("3. Make TSNE visualizations")
    print(" @_@ it can take time @_@ \n")
    for data_name, (x, y) in transformed_test.items():
        print("--------------------------")
        print(":::: ", data_name, "representation")
        tsne_plot(x, y, save=True, title=data_name + "_representation")
    print("resulted figures are saved in", VISUALIZATIONS_PATH)

    print("\n4. evaluating the  different classifiers in different data representaions")
    print(":::: Basic Metric : Mean Square Error :::::")
    print("-------------------------------------------")
    # dataframe result
    eval_results = pd.DataFrame(columns=list(classifiers.keys()), index=list(transformed_train.keys()))
    # train save and evaluate
    for clf_name, clf in classifiers.items():
        print("-", clf_name, ':')
        for index_name, (x, y) in transformed_train.items():

            loaded_model = ClassifierHelpers.load(filename=clf_name + '_' + index_name)
            if loaded_model:

                mse = ClassifierHelpers.evaluate(loaded_model, transformed_test[index_name][0],
                                                 transformed_test[index_name][1])
            else:

                clf.fit(x, y)
                ClassifierHelpers.save(clf=clf, name=clf_name + '_' + index_name)
                mse = ClassifierHelpers.evaluate(clf, transformed_test[index_name][0], transformed_test[index_name][1])

            print('    *{} representation : mse={}'.format(index_name, mse))
            eval_results.loc[index_name, clf_name] = mse
            print("-------------------------------------------")
    # save the result as a csv file
    csv_results = os.path.join(RESULTS_PATH, 'evaluation_results.csv')
    eval_results.to_csv(csv_results)
    print("5. save Evaluations Results")
    print("=>", csv_results)

    if os.uname()[0] == 'Linux':
        os.system('libreoffice {}'.format(csv_results))
