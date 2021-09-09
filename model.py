import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

vect = CountVectorizer()

amazon = open('amazon_cells_labelled.txt')
imdb = open('imdb_labelled.txt')
yelp = open('yelp_labelled.txt')

sentences = []
labels = []

for line in amazon:
    line = line.rstrip()
    if (line[-1] == "0"):
        sentences.append(line.rstrip()[0:-3])
        labels.append(0)
    else:
        sentences.append(line.rstrip()[0:-3])
        labels.append(1)

for line in imdb:
    line = line.rstrip()
    if (line[-1] == "0"):
        sentences.append(line.rstrip()[0:-3])
        labels.append(0)
    else:
        sentences.append(line.rstrip()[0:-3])
        labels.append(1)

for line in yelp:
    line = line.rstrip()
    if (line[-1] == "0"):
        sentences.append(line.rstrip()[0:-3])
        labels.append(0)
    else:
        sentences.append(line.rstrip()[0:-3])
        labels.append(1)

# print(sentences)

vect.fit(sentences)
# print(len(vect.get_feature_names()))

# np.save('features.npy',vect.get_feature_names())

simple_train_dtm = vect.transform(sentences)
X = simple_train_dtm.toarray()
# print(simple_train_dtm)

y = np.array(labels)
# print(type(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# model = MultinomialNB()
# model.fit(X_train, y_train)

print(X_test[2][200:300])

# print(model.predict([X_test[2]]))

# with open('model.sav', 'wb') as f:
#     pickle.dump(model,f)