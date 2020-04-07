'''
Pour le fonctionnement du package kaggle, https://github.com/Kaggle/kaggle-api
'''

import sklearn, argparse, os, kaggle, glob, pandas
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from classifier import *
from classifier.LDA import LDAClassifer
from classifier.RF import RF
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit


def argument_parser():
    parser = argparse.ArgumentParser(description='Classification de feuille d arbre utilisant 6 methode de classification differentes.')
    parser.add_argument('--method', type=str, default='MLP',
                         help='Permet d utiliser la methode specifie ou bien tous les faire.', choices=['MLP','regression','SVM','randomforest','naive_bayes', 'linear_discriminant_analysis', 'all'])
    parser.add_argument('--hidden_layer', type=tuple, default=(20,))
    return parser.parse_args()

def createDataSets():
    #Downloading leaf dataset from kaggle if not found.
    if not os.path.exists('data/train.csv'):
        os.chdir('./data')
        os.system('kaggle competitions download -c leaf-classification')
        with ZipFile('leaf-classification.zip','r') as zipObj:
            zipObj.extractall()
        with ZipFile('train.csv.zip','r') as zipObj:
            zipObj.extractall()
        with ZipFile('test.csv.zip','r') as zipObj:
            zipObj.extractall()
        for f in glob.glob('*.zip'):
            os.remove(f)
        os.chdir('..')

    train = pandas.read_csv('data/train.csv')
    test = pandas.read_csv('data/test.csv')

    data = LabelEncoder().fit(train.species)
    labels = data.transform(train.species)
    classes = list(data.classes_)
    test_ids = test.id

    train = train.drop(['species','id'], axis=1)
    test = test.drop(['id'], axis=1)

    return train, labels, test, test_ids, classes


if __name__ == '__main__':
    args = argument_parser()
    method = args.method

    train, labels, test, test_ids, classes = createDataSets()

    if method == 'MLP':
        hidden_layer = args.hidden_layer
        mlp = MLP(train, test, labels, test_ids, classes)

    elif method == 'regression':
        pass
    elif method == 'SVM':
        pass
    elif method == 'randomforest':
        rf_classifier = RF(train, labels, test, test_ids, classes)
        rf_classifier.hyperparam_search()
        rf_classifier.train()
        print(f'Justesse d\'entrainement: {rf_classifier.get_training_accuracy():%}')
        print(f'Justesse de validation: {rf_classifier.get_validation_accuracy():%}')
    elif method == 'adaboost':
        pass
    elif method == 'naive_bayes':
        nb_classifier = NBClassifer(train, labels, test, test_ids, classes)
        nb_classifier.search_hyperparameters()
        nb_classifier.train()
        print(f'Justesse d\'entrainement: {nb_classifier.get_training_accuracy():.2%}')
        print(f'Justesse de validation: {nb_classifier.get_validation_accuracy():.2%}')
    elif method == 'linear_discriminant_analysis':
        lda_classifier = LDAClassifer(train, labels, test, test_ids, classes)
        lda_classifier.search_hyperparameters()
        lda_classifier.train()
        print(f'Justesse d\'entrainement: {lda_classifier.get_training_accuracy():.2%}')
        print(f'Justesse de validation: {lda_classifier.get_validation_accuracy():.2%}')
    elif method == 'all':
        pass
    else:
        raise Exception('not a valid method')
