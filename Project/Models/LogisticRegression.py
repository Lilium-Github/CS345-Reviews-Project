from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import numpy as np



class LogisticRegressionTrainer:
    def __init__(self, max_features=1000, C=1, max_iter=1000):
        self.max_features = max_features
        self.C = C
        self.max_iter = max_iter
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words='english')
        self.model = None

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver='saga',
            n_jobs=-1
        )
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X):
        return self.model.predict(X)

    def results(self, X_test, y_test):
        train_preds = self.predict(self.X_train)
        test_preds  = self.predict(X_test)
        train_acc   = accuracy_score(self.y_train, train_preds) * 100
        test_acc    = accuracy_score(y_test, test_preds) * 100
        print(f"Train: {train_acc:.2f}%")
        print(f"Test:  {test_acc:.2f}%")
        return train_acc, test_acc

    def grid_search(self, X_train, y_train, X_test, y_test, param_grid={'C': [0.1, 1, 10]}):
        self.X_train = X_train
        self.y_train = y_train

        grid = GridSearchCV(
            LogisticRegression(max_iter=self.max_iter, solver='saga', n_jobs=-1),
            param_grid,
            cv=5,
            verbose=2,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        self.model = grid.best_estimator_
        train_acc  = accuracy_score(y_train, self.model.predict(X_train)) * 100
        test_acc   = accuracy_score(y_test,  self.model.predict(X_test))  * 100

        print(f"Best params: {grid.best_params_}")
        print(f"CV accuracy: {grid.best_score_}")
        print(f"Train: {train_acc:.2f}%")
        print(f"Test:  {test_acc:.2f}%")
        return grid.best_params_, grid.best_score_, train_acc, test_acc