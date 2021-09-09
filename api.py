from flask import Flask, json, request, jsonify, make_response
from flask_restful import Resource, Api, reqparse
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
api = Api(app)

vect = CountVectorizer()
features = np.load('features.npy')

with open("model.sav", 'rb') as file:  
    model = pickle.load(file)

class Predict(Resource):
    
    def post(self):        
        json_data = request.get_json(force=True)
        text = json_data['text']

        vect.fit([text])
        test_features = vect.get_feature_names()

        test_data = [0 for i in range(0,len(features))]

        for word in test_features:
            for i in range(0,len(features)):
                if word == features[i]:
                    test_data[i] += 1

        
        if model.predict([test_data]) == 1:
            return jsonify(message="sentence is positive :)")
        else:
            return jsonify(message="sentence is negative :(")

api.add_resource(Predict,'/predict')

if __name__ == '__main__':
    app.run()