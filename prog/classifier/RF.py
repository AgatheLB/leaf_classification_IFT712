from sklearn.ensemble import RandomForestClassifier
from classifier.classifier import Classifier


class RF(Classifier):
    """
    Classifieur - Random Forest
    """

    def __init__(self, train, labels, test, test_ids, classes):
        """
        :param train: Jeu d'entrainement, sera subdivisé pour valider l'entrainement
        :param labels: annotations
        :param test: Données à classifier
        :param test_ids: id de la dataframe de test pour le jeu de données leaf-classification
        :param classes: noms des espèces végétales
        """
        super(RF, self).__init__(train, labels, test, test_ids, classes)
        self.name = RandomForestClassifier.__name__
        self._classifier = RandomForestClassifier(n_jobs=-1)
        self._param_grid = {'n_estimators': [350, 400, 450], "max_depth": [20, 25, 30, 35]}
