import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

text_arr = []
rating_arr = []
counter = 100000

file = "Software.jsonl"
with open(file, 'r') as fp:
    for line in fp:
        line_dict = json.loads(line.strip())
        text_arr.append(line_dict['title'] + " " + line_dict['text'])
        rating_arr.append(line_dict['rating'])

        counter -= 1
        if counter == 0: break # this line is just here to make sure nothing breaks from a dataset too big
        
X = np.array(text_arr)
y = np.array(rating_arr)

model = make_pipeline(TfidfVectorizer(), MultinomialNB(alpha=0, force_alpha=True))

model.fit(X,y)

example_text = "hated this. 0 stars if i could"

prediction = model.predict([example_text])[0]

print("I think the following review:\n", example_text, "\nGave the item a rating of:", prediction)

print()

example_text = "New maid spends more time yowling at me and scrolling tumblr than cleaning."

prediction = model.predict([example_text])[0]

print("I think the following review:\n", example_text, "\nGave the item a rating of:", prediction)
