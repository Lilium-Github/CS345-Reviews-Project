from Amazon_Reviews import get_data, load_data
from Preprocessor import preprocess_ratings

# delete data_cache.pkl if you want to load a different file, otherwise it will load the cached data from the previous file
# set a limit to the number of records to load, or set to a very high number to load all records
# load_data() opens explorer to choose JSONL file with reviews and ratings

LIMIT = 500000 
load_data(LIMIT) 

# get_data() gets the data as the review text and rating only
data = get_data() 

# preprocess_ratings(data) removes stop words, punctuation, and HTML tags and splits into a map of ratings to reviews
X, y = preprocess_ratings(data)

# stored in sparse matrix for efficiency

print(X[:10].toarray())
print(X.shape)

