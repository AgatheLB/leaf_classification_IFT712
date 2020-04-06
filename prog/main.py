import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import argparse
from data_handler import LCDataset
import pandas as pd


def argument_parser():
    parser = argparse.ArgumentParser(description='Classification de feuille d arbre utilisant 6 methode '
                                                 'de classification differentes.')
    parser.add_argument('-m', '--method', type=str,
                        help='Permet d,utiliser la methode specifie.',
                        choices=['MLP','regression','SVM','randomforest','adaboost', 'linear_discriminant_analysis'],
                        required=True)
    parser.add_argument('-i', '--train_path', type=str, help='Chemin du jeu de données d\'entrainement', required=True)
    parser.add_argument('-t', '--test_path', type=str, help='Chemin du jeu de données de test', required=True)

    parser = parser.parse_args()

    return parser


def main():

    args = argument_parser()

    dataset = LCDataset(args.train_path, args.test_path)
    method = args.method

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
    elif method == 'linear_discriminant_analysis':
        linear_discriminant_analysis(dataset)
    else:
        raise Exception('not a valid method')


def linear_discriminant_analysis():
    pass

if __name__ == "__main__":
    main()