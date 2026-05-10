from sklearn.naive_bayes import ComplementNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from scipy.sparse import issparse
import numpy as np

class NaiveBayesTrainer:
    def __init__(self):
        self.model = None
        self.baseline = None

    def _remove_vader(self, X):
        if issparse(X):
            X = X.toarray()
        return X[:, 4:]  # skip first 4 VADER columns

    def train(self, X_train, y_train):
        self.y_train = y_train
        self.X_train = self._remove_vader(X_train)

        self.model = ComplementNB(alpha=1.0)
        self.model.fit(self.X_train, y_train)

        self.baseline = DummyClassifier(strategy='most_frequent')
        self.baseline.fit(self.X_train, y_train)

    def predict(self, X):
        return self.model.predict(self._remove_vader(X))

    def results(self, X_test, y_test):
        X_test_clean   = self._remove_vader(X_test)
        test_preds     = self.model.predict(X_test_clean)
        train_preds    = self.model.predict(self.X_train)
        baseline_preds = self.baseline.predict(X_test_clean)

        train_acc    = accuracy_score(self.y_train, train_preds) * 100
        test_acc     = accuracy_score(y_test, test_preds) * 100
        baseline_acc = accuracy_score(y_test, baseline_preds) * 100

        print(f"Train:    {train_acc:.2f}%")
        print(f"Test:     {test_acc:.2f}%")
        print(f"Baseline: {baseline_acc:.2f}%")
        print(f"Improvement over baseline: {test_acc - baseline_acc:.2f}%")
        return train_acc, test_acc, baseline_acc