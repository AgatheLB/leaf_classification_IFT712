import sklearn
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit


class LDAClassifer():
    """Classificateur - Analyse du Discriminante Linéaire
    """
    def __init__(self, train, labels, test, test_ids, classes):
        """
        :param train: Jeu d'entrainement, sera subdivisé pour valider l'entrainement
        :param labels: annotations
        :param test: Données à classifier
        :param test_ids: id de la dataframe de test pour le jeu de données leaf-classification
        :param classes: noms des espèces végétales
        """
        self._name = LinearDiscriminantAnalysis.__name__
        self._train = train
        self._test = test
        self._labels = labels
        self._test_ids = test_ids
        self._classes = np.array(classes)

        # "Single value decomposition" ne calcul pas de matrice de covariance, ce qui est bon
        # en l'occurence vu le grand nombre de dimensions
        self._classifier = LinearDiscriminantAnalysis(solver='svd')
        self._splitted_data = self._split_data()


    def _split_data(self):
        """
        Divise le jeux de données annoté en sous-ensembles d'entrainement et de test
        :return: Les 4 sous-ensembles X_train, Y_train, X_test, Y_test
        """
        # La stratification est souhaitable vu le grand nombre de classes. On s'assure que toutes
        # les classes soient représentées dans chaque sous-ensemble de données
        stratified_split = StratifiedShuffleSplit(n_splits=10, train_size=None, test_size=0.2,
                                                  random_state=37)

        for train_i, test_i in stratified_split.split(self._train, self._labels):
            X_train, X_test = self._train.values[train_i], self._train.values[test_i]
            Y_train, Y_test = self._labels[train_i], self._labels[test_i]

        return np.array([X_train, Y_train, X_test, Y_test])

    def train(self):
        """
        Entraine le modèle avec les jeux de données fournis à l'instanciation
        """
        x_train = self._splitted_data[0]
        y_train = self._splitted_data[1]
        self._classifier.fit(x_train, y_train)
        print(f'{self._name} trained!')

    def predict(self, x_predict, text_predictions=False):
        """predict
        :param x_predict: Jeu de données à classifier
        :param text_predictions: obtenir la classification en text (au lieu des int)
        :return: La classification pour chaque élments de x_predict, en chiffre ou en text selon text_prediction
        """
        if text_predictions:
            return self._classes[self._classifier.predict(x_predict)]
        else:
            return self._classifier.predict(x_predict)

    def validation(self):
        """validation
        :return: La justesse d'entrainement
        """
        prediction = self.predict(self._splitted_data[2])
        return sklearn.metrics.accuracy_score( self._splitted_data[3], prediction)
