import sklearn
import numpy as np
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB

class NBClassifer():
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
        self.name = GaussianNB.__name__
        self._train = train
        self._test = test
        self._labels = labels
        self._test_ids = test_ids
        self._classes = np.array(classes)

        self._hyperparameters = np.zeros(3, dtype='<U5')

        self._classifier = GaussianNB()
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
        self._classifier.fit(self._splitted_data[0], self._splitted_data[1])

    def predict(self, x_predict, text_predictions=False):
        """predict
        :param x_predict: Data to classify
        :param text_predictions: Obtain text prediction (instead of int)
        :return: La classification pour chaque élments de x_predict, en chiffre ou en text selon text_prediction
        """
        if text_predictions:
            return self._classes[self._classifier.predict(x_predict)]
        else:
            return self._classifier.predict(x_predict)

    def get_validation_accuracy(self):
        """validation
        :return: La justesse d'entrainement
        """
        prediction = self.predict(self._splitted_data[2])
        return sklearn.metrics.accuracy_score( self._splitted_data[3], prediction)

    def get_training_accuracy(self):
        """validation
        :return: La justesse d'entrainement
        """
        prediction = self.predict(self._splitted_data[0])
        return sklearn.metrics.accuracy_score( self._splitted_data[1], prediction)

    def search_hyperparameters(self):
        variance_smoothing = [1e-11, 1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-04, 1e-3]
        _max_score = -np.inf
        _counter = 0
        _hyperparameters = 0
        for var_smoothing in variance_smoothing:
            self._classifier = GaussianNB(var_smoothing=var_smoothing)
            self.train()
            _score = self.get_validation_accuracy()
            if _score > _max_score:
                _max_score = _score
                _hyperparameters = var_smoothing
                print(f'hyperparameters updated - variance smoothing: {var_smoothing}')
            _counter += 1
            print(f'Searching hyperparameters for {self.name}, iteration: '
                  f'{_counter}/{len(variance_smoothing)} '
                  f'score: {_score:%}, max score: {_max_score:%}')

        self._classifier = GaussianNB(var_smoothing=_hyperparameters)
        self.train()
