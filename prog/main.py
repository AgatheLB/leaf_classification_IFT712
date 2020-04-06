'''
Pour le fonctionnement du package kaggle, https://github.com/Kaggle/kaggle-api
'''

import sklearn, argparse, os, kaggle
from zipfile import ZipFile

def argument_parser():
    parser = argparse.ArgumentParser(description='Classification de feuille d arbre utilisant 6 methode de classification differentes.')
    parser.add_argument('--method', type=str, default='MLP',
                         help='Permet d,utiliser la methode specifie.', choices=['MLP','regression','SVM','randomforest','adaboost'])
    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()

    method = args.method

    #Downloading leaf dataset from kaggle.
    if not os.path.exists('data/train.csv'):
        os.chdir('./data')
        os.system('kaggle competitions download -c leaf-classification')
        with ZipFile('leaf-classification.zip','r') as zipObj:
            zipObj.extractall()
        os.remove('leaf-classification.zip')
        os.chdir('..')

    

    if method == 'MLP':
        pass
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
    else:
        raise Exception('not a valid method')