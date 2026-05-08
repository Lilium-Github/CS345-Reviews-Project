from Amazon_Reviews import get_data, load_data
from Preprocessor import preprocess_ratings
from Models.Torch_Feedforward import FeedForwardTrainer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# delete data_cache.pkl if you want to load a different file, otherwise it will load the cached data from the previous file
# set a limit to the number of records to load, or set to a very high number to load all records
# load_data() opens explorer to choose JSONL file with reviews and ratings

LIMIT = 100000 
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


# Feed forward (with gpu) 

# Attempt #1 100,000 reviews Train: 71.169% Test: 70.54% hidden_size=256, batch_size=256, epochs=15 and dropout=0.5, TFIDF_FEATURES=1000 Data: VideoGames 

# Attempt #2 100,000 reviews Train: 64.09% Test: 63.07% hidden_size=256, batch_size=256, epochs=15 and dropout=0.5, TFIDF_FEATURES=1000 Data: Software 

# Attempt #3 100,000 reviews Train: 75.11% Test: 74.11% hidden_size=256, batch_size=256, epochs=15 and dropout=0.2, TFIDF_FEATURES=1000 Data: Office_Products 
y_remapped = y - 1
X_train, X_test, y_train, y_test = train_test_split(X, y_remapped, test_size=0.1, random_state=42)

results = []

for hidden_size in [128, 256]:
    for dropout in [0.2, 0.3, 0.5, 0.6]:
        for lr in [0.001, 0.01,0.1]:
            trainer = FeedForwardTrainer(
                hidden_size=hidden_size,
                epochs=15,
                batch_size=64,
                dropout=dropout,
                lr=lr
            )
            trainer.train(X_train, y_train)
            train_predictions = trainer.predict(X_train)
            predictions = trainer.predict(X_test)
         
            acc = accuracy_score(y_test, predictions)
            results.append((acc, hidden_size, dropout, lr))
            train_acc = accuracy_score(y_train, train_predictions)
            test_acc = accuracy_score(y_test, predictions)
            print("train: ", train_acc, " test:", test_acc)

results.sort(reverse=True)
print("\nBest:", results[0])