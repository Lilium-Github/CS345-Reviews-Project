import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
import nltk
# nltk.download('vader_lexicon')

class VaderBayes:
    def __init__(self):  
        # step 6: set everything up for the pipeline
        text_transformer = CountVectorizer(stop_words='english')

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

# step 1: extract data from the json file
text_arr = []
rating_arr = []
counter = 1000

file = "Software.jsonl"
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
X = df[['text', 'vader_score']]
y = df['label']


# step 5: train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

# step 8: train it
model = VaderBayes()
model.fit(X_train, y_train)

# step 9: predict
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")
