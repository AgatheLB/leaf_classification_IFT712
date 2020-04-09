import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


class Classifier:
    """
    Classe parente pour différent Classificateur. Elle implemente les fonctions recherche d'hyperparamètre, d'entrainement
    et des fonctions de présentation des résultats.
    """
    def __init__(self, train, labels, test, test_ids, classes):
        """
        :param train: Jeu d'entrainement, sera subdivisé pour valider l'entrainement
        :param labels: annotations
        :param test: Données à classifier
        :param test_ids: id de la dataframe de test pour le jeu de données leaf-classification
        :param classes: noms des espèces végétales
        """
        self._train = train
        self._test = test
        self._labels = labels
        self._test_ids = test_ids
        self._classes = np.array(classes)

        self._X_train, self._y_train, self._X_valid, self._y_valid = self._split_data()

        self._best_model = None
        self._best_pair = None

    def _split_data(self):
        """
        Divise le jeu de données annotées en sous-ensembles d'entrainement et de validation
        :return: Les 4 sous-ensembles X_train, y_train, X_valid, y_valid
        """
        stratified_split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=37)

        for index_train, index_valid in stratified_split.split(self._train, self._labels):
            X_train, X_valid = self._train.values[index_train], self._train.values[index_valid]
            y_train, y_valid = self._labels[index_train], self._labels[index_valid]

        return X_train, y_train, X_valid, y_valid

    def search_hyperparameters(self):
        """
        Entreprend une recherche d'hyper-paramètres. Le meilleur modèle trouvé est sauvegardé dans self._best_model et
        les meilleurs hyper-paramètres trouvés dans self._best_pair
        """
        grid = GridSearchCV(self._classifier, self._param_grid, scoring='accuracy', n_jobs=-1, verbose=1)
        grid.fit(self._X_train, self._y_train)

        self._best_model = grid
        self._best_pair = grid.best_params_
        print(f'Meilleurs paramètres trouvés pour {self.name} sont {self._best_pair} pour une justesse de '
              f'{grid.best_score_:.2%}')

    def train(self):
        """
        Entraîne le modèle avec le jeu de données fournis à l'instanciation
        """
        self._classifier.set_params(**self._best_pair)
        self._classifier.fit(self._X_train, self._y_train)

        print(f'{self.name} trained avec les paramètres {self._best_pair}')

    def get_training_accuracy(self):
        """
        Calcule la justesse d'entraînement
        :return: La justesse d'entrainement
        """
        return accuracy_score(self._y_train, self._classifier.predict(self._X_train))

    def get_validation_accuracy(self):
        """
        Calcule la justesse de validation
        :return: La justesse de validation
        """
        return accuracy_score(self._y_valid, self._classifier.predict(self._X_valid))

    def display_accuracies(self):
        print(f'Justesse d\'entrainement: {self.get_training_accuracy():.2%}')
        print(f'Justesse de validation: {self.get_validation_accuracy():.2%}')