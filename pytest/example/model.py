# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2025/05/16 07:47:22
@Author  :   orange 
@Version :   1.0
@Contact :   chenorange2219@gmail.com
'''


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


class MLPredictor:
    def __init__(self):
        self.model = None

    def train(self):
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )
        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)

    def predict(self, data):
        if not self.model:
            raise Exception("Model is not trained yet!")
        return self.model.predict([data])

    def save_model(self, path="model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, path="model.pkl"):
        with open(path, "rb") as f:
            self.model = pickle.load(f)