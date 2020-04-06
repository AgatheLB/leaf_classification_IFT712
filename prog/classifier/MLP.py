from sklearn.neural_network import MLPClassifier

class MLP():
    def __init__(self, train, test, labels, test_ids, classes):
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
                    model = MLPClassifier(hl, learning_rate_init=lr, solver=optim)
                    fit_model = model.fit(self.train, self.labels)

                    pred = fit_model.predict(self.test)

                    

