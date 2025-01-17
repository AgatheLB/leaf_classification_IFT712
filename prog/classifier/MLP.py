from sklearn.neural_network import MLPClassifier
from classifier.classifier import Classifier

class MLP(Classifier):
    """
    Classificateur - MultiLayer Perceptron
    """
    def __init__(self, train, labels, test, test_ids, classes):
        """
        :param train: Jeu d'entrainement, sera subdivisé pour valider l'entrainement
        :param labels: Annotations
        :param test: Données à classifier
        :param test_ids: Id de la dataframe de test pour le jeu de données leaf-classification
        :param classes: Noms des espèces végétales
        """
        super(MLP, self).__init__(train, labels, test, test_ids, classes)
        self.name = MLPClassifier.__name__
        self._classifier = MLPClassifier()
        self._param_grid = {'hidden_layer_sizes': [(50,), (80,), (100,)],
                            'learning_rate_init': [1e-1, 1e-2, 1e-3],
                            'solver': ['adam', 'sgd'],
                            'activation': ['relu', 'logistic']}
