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
from classifier.KNN import KNN
from classifier.Regression import Regression
from sklearn.preprocessing import LabelEncoder

def argument_parser():
    parser = argparse.ArgumentParser(description='Classification de feuille d arbre utilisant 6 methode de classification differentes.')
    parser.add_argument('--method', type=str, default='MLP',
                        help='Permet d utiliser la methode specifie ou bien tous les faire.',
                        choices=['MLP','regression','SVM','randomforest','naive_bayes','linear_discriminant_analysis',
                                 'KNN', 'all'])
    parser.add_argument('--hidden_layer', type=tuple, default=(20,))
    return parser.parse_args()

def createDataSets():
    """
    Fonction permettant de télécharger l'ensemble de donnée leaf-classification du
    site web de Kaggle et de créer un ensemble de données d'entrainement et de validation

    return: train: ensemble de données pour l'entrainement Nx3x64
    return: labels: vecteur N x 1 contenant le numero de classe de feuille.
    return: test: ensemble de données pour la validation Nx3x64
    return: test_id: ensemble des identifiants des individus dans l'ensemble test
    return: classes: ensemble de toutes les classes possibles Cx1
    """
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
    classes = np.array(data.classes_)
    test_ids = test.id

    train = train.drop(['species','id'], axis=1)
    test = test.drop(['id'], axis=1)

    return train, labels, test, test_ids, classes


if __name__ == '__main__':
    args = argument_parser()
    method = args.method

    train, labels, test, test_ids, classes = createDataSets()
    classifiers = []

    if method == 'MLP':
        clf = MLP(train, labels, test, test_ids, classes)
        classifiers.append(clf)
    elif method == 'regression':
        clf = Regression(train, labels, test, test_ids, classes)
        classifiers.append(clf)
    elif method == 'SVM':
        clf = SVM(train, labels, test, test_ids, classes)
        classifiers.append(clf)
    elif method == 'randomforest':
        clf = RF(train, labels, test, test_ids, classes)
        classifiers.append(clf)
    elif method == 'KNN':
        clf = KNN(train, labels, test, test_ids, classes)
        classifiers.append(clf)
    elif method == 'naive_bayes':
        clf = NB(train, labels, test, test_ids, classes)
        classifiers.append(clf)
    elif method == 'linear_discriminant_analysis':
        clf = LDA(train, labels, test, test_ids, classes)
        classifiers.append(clf)
    elif method == 'all':
        clfs = [MLP, SVM, RF, KNN, NB, LDA]
        for clf in clfs:
            classifier = clf(train, labels, test, test_ids, classes)
            classifiers.append(classifiers)
    else:
        raise Exception('not a valid method')

    for c in classifiers:
        c.search_hyperparameters()
        c.train()
        c.display_accuracies()
