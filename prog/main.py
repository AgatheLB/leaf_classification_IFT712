import sklearn
import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description='Classification de feuille d arbre utilisant 6 methode de classification differentes.')
    parser.add_argument('--method', type=str,
                         description='Permet d,utiliser la methode specifie.', choices=['MLP','regression','SVM','randomforest','adaboost'])
    return parser


if __name__ == "__main__":
    args = argument_parser()

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
    elif method == '':
        pass
    else:
        raise Exception('not a valid method')