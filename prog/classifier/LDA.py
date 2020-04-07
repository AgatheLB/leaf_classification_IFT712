import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from classifier.classifier import Classifier


class LDA(Classifier):
    """
    Classificateur - Analyse du Discriminante Linéaire
    """

    def __init__(self, train, labels, test, test_ids, classes):
        """
        :param train: Jeu d'entrainement, sera subdivisé pour valider l'entrainement
        :param labels: annotations
        :param test: Données à classifier
        :param test_ids: id de la dataframe de test pour le jeu de données leaf-classification
        :param classes: noms des espèces végétales
        """
        Classifier.__init__(self, train, labels, test, test_ids, classes)
        self.name = LinearDiscriminantAnalysis.__name__
        self._classifier = LinearDiscriminantAnalysis()
        self._param_grid = {'solver': ['svd', 'lsqr', 'eigen'],
                            'n_components': np.int_(
                                np.ceil(np.linspace(self._classes.shape[0] - 10, self._classes.shape[0] - 1))),
                            'tol': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]}
