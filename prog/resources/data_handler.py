import pandas as pd

from sklearn.preprocessing import LabelEncoder

class LCDataset():
    def __init__(self, train_path, test_path):
        self._train_path = train_path
        self._test_path = test_path

        self.train = pd.read_csv(self._train_path)
        self.test = pd.read_csv(self._test_path)

        self._preprocess_data()

    # def __len__(self):
    #
    # def __getitem__(self, item):
    #     pass
    def _preprocess_data(self):
        le = LabelEncoder()
        le.fit(self.train.loc[:, 'species'])
        classes = list(le.classes_)