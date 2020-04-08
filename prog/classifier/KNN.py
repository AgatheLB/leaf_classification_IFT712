from sklearn.neighbors import KNeighborsClassifier
from classifier.classifier import Classifier


class KNN(Classifier):
    """
    Classificateur - Plus proche voisin
    """

    def __init__(self, train, labels, test, test_ids, classes):
        """
        :param train: Jeu d'entrainement, sera subdivisé pour valider l'entrainement
        :param labels: annotations
        :param test: Données à classifier
        :param test_ids: id de la dataframe de test pour le jeu de données leaf-classification
        :param classes: noms des espèces végétales
        """
        super(KNN, self).__init__(train, labels, test, test_ids, classes)
        self.name = KNeighborsClassifier.__name__
        self._classifier = KNeighborsClassifier()
        self._param_grid = {'n_neighbors': [1, 2, 3, 4, 5],
                            'weights': ['uniform', 'distance'],
                            'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
                            'leaf_size': [10, 20, 30, 40, 50],
                            'p': [1, 2]}
