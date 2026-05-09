from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


def Svm_Trainer(X_train, X_test, y_train, y_test):

    param_grid = {
        'C': [0.1, 1, 10],
        
    }

    grid = GridSearchCV(SVC(kernel='linear'), param_grid, cv=5, verbose=2,n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print("CV accuracy:", grid.best_score_)

    best_svm = grid.best_estimator_
    train_predictions = best_svm.predict(X_train)
    test_predictions = best_svm.predict(X_test)

    train_acc = accuracy_score(y_train, train_predictions)
    test_acc = accuracy_score(y_test, test_predictions)

    print(f"Train: {train_acc*100:.2f}%")
    print(f"Test: {test_acc*100:.2f}%")

    return best_svm, grid.best_params_
   
    
