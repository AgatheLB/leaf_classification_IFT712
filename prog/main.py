'''
Pour le fonctionnement du package kaggle, https://github.com/Kaggle/kaggle-api
'''

import sklearn, argparse, os, kaggle, glob, pandas
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from classifier.NB import NB
from classifier.LDA import LDA
from classifier.RF import RF
from classifier.MLP import MLP
from classifier.SVM import SVM
from sklearn.preprocessing import LabelEncoder



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
        classifier = MLP(train, labels, test, test_ids, classes)
    elif method == 'regression':
        pass
    elif method == 'SVM':
        classifier = SVM(train, labels, test, test_ids, classes)
    elif method == 'randomforest':
        classifier = RF(train, labels, test, test_ids, classes)
    elif method == 'adaboost':
        pass
    elif method == 'naive_bayes':
        classifier = NB(train, labels, test, test_ids, classes)
    elif method == 'linear_discriminant_analysis':
        classifier = LDA(train, labels, test, test_ids, classes)
    elif method == 'all':
        pass
    else:
        raise Exception('not a valid method')
    classifier.search_hyperparameters()
    classifier.train()
    classifier.display_accuracies()
