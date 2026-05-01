import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
import nltk
# nltk.download('vader_lexicon')

class VaderBayes:
    def __init__(self):  
        # step 6: set everything up for the pipeline
        text_transformer = TfidfVectorizer(stop_words='english')

        vader_pipeline = Pipeline([
            ('reshape', FunctionTransformer(lambda x: x.values.reshape(-1, 1), validate=False)),     # bayes doesn't like negatives, fixing that here
            ('scale', MinMaxScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('text', text_transformer, 'text'),
                ('vader', vader_pipeline, 'vader_score')
            ]
        )

        # step 7: create the naive bayes model
        self.model = Pipeline([
            ('features', preprocessor),
            ('classifier', MultinomialNB())
        ])

    def fit(self, X, y):
        self.model.fit(X,y)
    
    def predict(self, X):
        return self.model.predict(X)

# _______________________________________________________________________________________

files = ["Office_Products.jsonl", "Video_Games.jsonl", "Software.jsonl"]
counters = range(1,100)

means = []
vader_means = []

for f in files:
    predictions = []
    vader_predictions = []

    # step 1: extract data from the json file
    text_arr = []
    rating_arr = []
    counter = 10000

    file = f
    with open(file, 'r') as fp:
        for line in fp:
            line_dict = json.loads(line.strip())
            text_arr.append(line_dict['title'] + " " + line_dict['text'])
            rating_arr.append(line_dict['rating'])

            counter -= 1
            if counter == 0: break # this line is just here to make sure nothing breaks from a dataset too big
            

    # step 2: create a pd.dataframe 'df' from that
    data = {
        'text': text_arr,
        'label': rating_arr
    }
    df = pd.DataFrame(data)

    # step 3: add a vader_score column to the dataframe
    sia = SentimentIntensityAnalyzer()
    df['vader_score'] = df['text'].apply(lambda x: sia.polarity_scores(x)['compound'])

    # step 4: combine everything into X and y
    X = df['text']
    X_vader = df[['text', 'vader_score']]
    y = df['label']

    for c in counters:

        # step 5: train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=c)
        X_vader_train, X_vader_test, y_vader_train, y_vader_test = train_test_split(X_vader, y, test_size=0.25, random_state=c)

        # step 8: train it
        model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
        model.fit(X_train, y_train)

        vader_model = VaderBayes()
        vader_model.fit(X_vader_train, y_vader_train)

        # step 9: predict
        prediction = model.predict(X_test)
        #print(f"Falsetto Algorithm Accuracy: {accuracy_score(y_test, prediction)}")
        predictions.append(accuracy_score(y_test, prediction))

        vader_prediction = vader_model.predict(X_vader_test)
        #print(f"Serra Algorithm Accuracy: {accuracy_score(y_vader_test, vader_prediction)}")
        vader_predictions.append(accuracy_score(y_vader_test, vader_prediction))

    print(f"-----------------------------------------------------------------\nData: {f}")
    print(f"Naive Bayes on its own got an average of {np.mean(predictions)}")
    means.append(np.mean(predictions))
    print(f"Bayes + Vader got an average of {np.mean(vader_predictions)}")
    vader_means.append(np.mean(vader_predictions))

models = ("Naive Bayes", "Bayes with VADER")
data_means = {
    files[0]: (means[0], vader_means[0]),
    files[1]: (means[1], vader_means[1]),
    files[2]: (means[2], vader_means[2]),
}

x = np.arange(len(model))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in data_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Mean Accuracy')
ax.set_title('Dataset performance with stopwords')
ax.set_xticks(x + width, models)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0.4, 1)

plt.show()