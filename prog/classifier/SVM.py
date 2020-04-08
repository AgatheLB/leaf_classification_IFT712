from sklearn.svm import SVC
from classifier.classifier import Classifier


class SVM(Classifier):
    """
    Classifieur - Machine à vecteurs de support
    """

    def __init__(self, train, labels, test, test_ids, classes):
        """
        :param train: Jeu d'entrainement, sera subdivisé pour valider l'entrainement
        :param labels: annotations
        :param test: Données à classifier
        :param test_ids: id de la dataframe de test pour le jeu de données leaf-classification
        :param classes: noms des espèces végétales
        """
        super(SVM, self).__init__(train, labels, test, test_ids, classes)
        self.name = SVC.__name__
        self._classifier = SVC()
        self._param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                      'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
