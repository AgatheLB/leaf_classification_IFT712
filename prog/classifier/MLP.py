from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

import numpy as np

class MLP():
    '''
    Multi-Layer Perceptron Wrapper Class
    '''
    def __init__(self, train, test, labels, test_ids, classes):
        self.best_score = 0
        self.best_pair = None
        self.best_model = None
        self.train = train
        self.test = test
        self.labels = labels
        self.test_ids = test_ids
        self.classes = classes
    
    def hyperparamSearch(self, k=5, hiddens=[(20,)], learning_rates=[1e-3]):
        '''
            takes the current classifier and try to find the best learning rate and solver pair with a k-fold algorithm.
        '''
        for hl in hiddens:
            for lr in learning_rates:
                for optim in ['sgd','adam']:
                    print('training for hidden layer: {} and learning rate:{:f} with {:s}'.format(hl,lr,optim))
                    score = 0
                    model = MLPClassifier(hl, learning_rate_init=lr, solver=optim, max_iter=1500)
                    fit_model = model.fit(self.train, self.labels)
                    score = fit_model.score(self.train, self.labels)
                    if self.best_score <= score:
                        self.best_model = fit_model
                        self.best_pair = (hl,lr,optim)
                        self.best_score = score
        
        return self.best_model, self.best_pair





