from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.svm as svm

class DummyMLModel:
    data =  pd.DataFrame(np.array([["good weather", "many", "Y", "67.45"],
                                 ["rainy", "many", "N", "30.42"],
                                 ["windy","many", "Y", "98.25"],
                                 ["rainy", "many", "N", "27.45"],
                                 ["windy", "many", "Y", "6.39"],
                                 ["good weather", "many", "N", "52.75"],
                                 ["rainy", "many", "Y", "19.36"],
                                 ["windy", "many", "N", "18.39"],
                                 ["rainy", "many", "Y", "17.32"],
                                 ["good weather", "many", "Y", "88.29"],
                                 ["good weather", "some", "N", "79.21"],
                                 ["windy", "many", "Y", "29.63"],
                                 ["rainy", "many", "Y", "46.28"],
                                 ["rainy", "some", "N", "76.43"],
                                 ["good weather", "many", "N", "19.32"],
                                 ["windy", "many", "Y", "30.42"],
                                 ["good weather", "some", "N", "42.75"],
                                 ["rainy", "many", "N", "30.03"],
                                 ["rainy", "many", "Y", "62.69"],
                                 ["good weather", "little", "N", "89.99"],
                                 ["windy", "little", "N", "106.43"],
                                 ["rainy", "many", "Y", "105.67"],
                                 ["good weather", "little", "N", "95.40"],

                                 ["good weather", "many", "Y", "71.45"],
                                 ["rainy", "some", "N", "74.42"],
                                 ["windy","many", "Y", "98.25"],
                                 ["rainy", "little", "N", "109.45"],
                                 ["windy", "many", "Y", "73.39"],
                                 ["good weather", "some", "N", "52.75"],
                                 ["rainy", "many", "Y", "90.36"],
                                 ["windy", "little", "N", "95.39"],
                                 ["rainy", "many", "Y", "101.32"],
                                 ["good weather", "many", "Y", "150.29"],
                                 ["good weather", "some", "N", "63.21"],
                                 ["windy", "many", "Y", "69.63"],
                                 ["rainy", "many", "Y", "72.28"],
                                 ["rainy", "some", "N", "76.43"],
                                 ["good weather", "some", "N", "66.32"],
                                 ["windy", "many", "Y", "43.42"],
                                 ["good weather", "some", "N", "42.75"],
                                 ["rainy", "many", "N", "39.03"],
                                 ["rainy", "many", "Y", "93.69"],
                                 ["good weather", "little", "N", "86.99"],
                                 ["windy", "little", "N", "146.43"],
                                 ["rainy", "many", "Y", "127.67"],
                                 ["good weather", "many", "N", "30.40"],
                                   ["good weather", "some", "N", "40.99"],
                                   ["windy", "many", "N", "39.43"],
                                   ["rainy", "many", "Y", "99.67"],
                                   ["good weather", "many", "Y", "100.40"],
                                   ["rainy", "many", "Y", "99.67"],
                                   ["good weather", "many", "Y", "100.40"],
                                   ["good weather", "little", "N", "86.99"],
                                   ["windy", "little", "N", "146.43"],
                                   ["rainy", "many", "Y", "127.67"],
                                   ["good weather", "many", "N", "30.40"],
                                   ["good weather", "little", "N", "86.99"],
                                   ["windy", "little", "N", "146.43"],
                                   ["rainy", "many", "Y", "127.67"],
                                   ["good weather", "many", "N", "30.40"],
                                   ["good weather", "little", "N", "83.99"],
                                   ["windy", "many", "Y", "101.43"],
                                   ["rainy", "little", "N", "103.99"],
                                   ["rainy", "many", "Y", "106.43"],
                                   ["good weather", "some", "N", "79.99"],
                                   ["windy", "many", "Y", "99.43"],
                                   ["rainy", "some", "N", "76.99"],
                                   ["rainy", "many", "Y", "98.43"],
                                   ["windy", "some", "N", "76.99"],
                                   ["good weather", "many", "Y", "95.43"],
                                   ["good weather", "some", "N", "79.99"],
                                   ["windy", "many", "Y", "99.43"],
                                   ]),
                        columns=["weather", "visitorsOnThisDay", "holidayYN", "ticketPrice"])

    def __init__(self):
        self.splitData()
        self.fitModel()

    def splitData(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data[["weather", "holidayYN", "ticketPrice"]],
            self.data["visitorsOnThisDay"],
            test_size=0.33, random_state=18)
        self.testdata = self.X_test
        self.testdata["visitorsOnThisDay"] = self.y_test

    def encodeData(self, data):
        if data.ndim == 2:
            data["weather"] = data["weather"].apply(self.weatherTransmission)
            data["holidayYN"] = data["holidayYN"].apply(self.holidayTransmission)
        if data.ndim == 1:
            data["weather"] = self.weatherTransmission(data["weather"])
            data["holidayYN"] = self.holidayTransmission(data["holidayYN"])

    def weatherTransmission(self, weatherState):
        switcher={
            "0": 0,
            "good weather": 1,
            "rainy":2,
            "windy":3
        }
        return switcher.get(weatherState, weatherState)

    def holidayTransmission(self, holidayState):
        switcher={
            "0": 0,
            "Y": 1,
            "N": 2
        }
        return switcher.get(holidayState, holidayState)

    def fitModel(self):
        self.encodeData(self.X_train)
        self.clf = svm.SVC(kernel="poly", probability=True, decision_function_shape='ovo', C=0.75)
        self.clf.fit(self.X_train, self.y_train)

    def predict(self, dataset):
        self.encodeData(dataset)
        dataset = dataset.drop("visitorsOnThisDay")
        dt = {'names': ['predicted', 'dist'], 'formats': [object, np.float]}
        prediction = np.zeros(len(self.data['visitorsOnThisDay'].unique()), dtype=dt)
        prediction['predicted'] = self.clf.classes_
        prediction['dist'] = self.clf.predict_proba(dataset.to_numpy().reshape(1,-1))
        prediction = prediction[np.argsort(prediction["dist"])]
        prediction = y = np.flipud(prediction)
        return prediction



