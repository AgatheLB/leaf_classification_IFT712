from sklearn.naive_bayes import GaussianNB
from classifier.classifier import Classifier


class NB(Classifier):
    """
    Classifieur - Bayes Naif
    """

    def __init__(self, train, labels, test, test_ids, classes):
        """
        :param train: Jeu d'entrainement, sera subdivisé pour valider l'entrainement
        :param labels: annotations
        :param test: Données à classifier
        :param test_ids: id de la dataframe de test pour le jeu de données leaf-classification
        :param classes: noms des espèces végétales
        """
        super(NB, self).__init__(train, labels, test, test_ids, classes)
        self.name = GaussianNB.__name__
        self._classifier = GaussianNB()
        self._param_grid = {'var_smoothing': [1e-11, 1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-04, 1e-3]}
