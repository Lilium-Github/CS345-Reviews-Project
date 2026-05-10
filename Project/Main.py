from Amazon_Reviews import get_data, load_data
from Preprocessor import preprocess_ratings
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# delete data_cache.pkl if you want to load a different file, otherwise it will load the cached data from the previous file
# set a limit to the number of records to load, or set to a very high number to load all records
# load_data() opens explorer to choose JSONL file with reviews and ratings

LIMIT = 10000 
load_data(LIMIT) 

# get_data() gets the data as the review text and rating only
data = get_data() 

# preprocess_ratings(data) removes stop words, punctuation, and HTML tags and splits into a map of ratings to reviews
X, y = preprocess_ratings(data, use_vader=True)

# stored in sparse matrix for efficiency

print(type(X))
print(type(y))
print(X[:20].toarray())
print("Matrix size: ", X.shape)

#Feedforward:
#from Models.Torch_Feedforward import FeedForwardTrainer
#y_remapped = y - 1
#X_train, X_test, y_train, y_test = train_test_split(X, y_remapped, test_size=0.1, random_state=42)
#trainer = FeedForwardTrainer()
#results = trainer.grid_search(X_train, y_train, X_test, y_test)

 #SVM:
from Models.SVM import Svm_Trainer
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Svm_Trainer(X_train, X_test, y_train, y_test)



#LogisticRegression:
#from Models.LogisticRegression import LogisticRegressionTrainer
#regtrainer = LogisticRegressionTrainer(max_features=3000)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#regtrainer.grid_search(X_train, y_train, X_test, y_test)


# Naive Bayes:
# from Models.Naive_Bayes import NaiveBayesTrainer
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# trainer = NaiveBayesTrainer()
# trainer.train(X_train, y_train)
# trainer.results(X_test, y_test)

#trainer.predict_example("hated this. 0 stars if i could")
#trainer.predict_example("New maid spends more time yowling at me and scrolling tumblr than cleaning.")


# (10000, 5004) on books

# Bayes
# Matrix size:  
# Train: 72.21%
# Test: 59.90%

# Logistic Regression
# CV accuracy: 0.6312499999999999
# Train: 77.50%
# Test: 62.50%

#SVM    
# Train: 81.91%
# Test: 63.00%

# Feedforward
# Train: 99.32%
# Test: 59.80%



  

