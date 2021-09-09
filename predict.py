from operator import le
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

vect = CountVectorizer()

with open("model.sav", 'rb') as file:  
    model = pickle.load(file)

features = np.load('features.npy')
# print(len(features))

sentance = input("Enter your sentence : ")

vect.fit([sentance])
test_features = vect.get_feature_names()
# print(test_features)

test_data = [0 for i in range(0,len(features))]

for word in test_features:
    for i in range(0,len(features)):
        if word == features[i]:
            test_data[i] += 1
            # print(i)


# simple_train_dtm = vect.transform(sentance)
# X = simple_train_dtm.toarray()

if model.predict([test_data]) == 1:
    print("sentence is positive :)")
else:
    print("sentence is negative :)")

# print(simple_train_dtm)