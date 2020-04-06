'''
Pour le fonctionnement du package kaggle, https://github.com/Kaggle/kaggle-api
'''

import sklearn, argparse, os, kaggle, glob, pandas
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from classifier.MLP import MLP
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

def argument_parser():
    parser = argparse.ArgumentParser(description='Classification de feuille d arbre utilisant 6 methode de classification differentes.')
    parser.add_argument('--method', type=str, default='MLP',
                         help='Permet d utiliser la methode specifie ou bien tous les faire.', choices=['MLP','regression','SVM','randomforest','adaboost', 'all'])
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
    classes = np.array(data.classes_)
    test_ids = test.id

    train = train.drop(['species','id'], axis=1)
    test = test.drop(['id'], axis=1)

    train = train.to_numpy()
    test = test.to_numpy()

    return train, labels, test, test_ids, classes


if __name__ == "__main__":
    args = argument_parser()
    method = args.method

    train, labels, test, test_ids, classes = createDataSets()

    if method == 'MLP':
        classifier = MLP(train, test, labels, test_ids, classes)
        
        hlayers = [ (10,),
                    (10,10),
                    (100,),
                    (100,100)
        ]
        lr = [  1e-3,
                2e-3,
                3e-3,
                1e-4,
                2e-4,
                5e-5
        ]

        mlp, hyperParams = classifier.hyperparamSearch(hiddens=hlayers,learning_rates=lr)
        print('meilleure hyper parametre trouver: {}'.format(hyperParams))
        test_pred_prob = mlp.predict_proba(test)
        
        rndIdx = np.random.randint(0, len(test_pred_prob))
        top10idx = test_pred_prob[rndIdx].argsort()[-10:][::-1]
        top10Classes = classes[top10idx]
        test = test_pred_prob[rndIdx][top10idx]
        plt.bar(top10Classes,test)
        plt.xticks(rotation=-45)
        plt.show()

    elif method == 'regression':
        pass
    elif method == 'SVM':
        pass
    elif method == 'randomforest':
        pass
    elif method == 'adaboost':
        pass
    elif method == '':
        pass
    elif method == 'all':
        pass
    else:
        raise Exception('not a valid method')