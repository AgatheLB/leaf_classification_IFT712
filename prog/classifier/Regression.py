from classifier.classifier import Classifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeClassifier

class Regression(Classifier):
    """
    Classificateur - Regression de Ridge
    """
    def __init__(self, train, labels, test, test_ids, classes):
        """
        :param train: Jeu d'entrainement, sera subdivisé pour valider l'entrainement
        :param labels: annotations
        :param test: Données à classifier
        :param test_ids: id de la dataframe de test pour le jeu de données leaf-classification
        :param classes: noms des espèces végétales
        """
        super(Regression,self).__init__(train, labels, test, test_ids, classes)
        self.name = RidgeClassifier.__name__
        self._classifier = RidgeClassifier()
        self._param_grid = {'alpha':[1e-1,2e-1,1e-2,1e-3,1e-4]}