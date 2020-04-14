from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.svm as svm

class DummyMLModel:
    data =  pd.DataFrame(np.array([["good weather", "many", "Y", "67.45"],
                                 ["rainy", "heaps", "N", "30.42"],
                                 ["windy","some", "Y", "98.25"],
                                 ["rainy", "heaps", "N", "27.45"],
                                 ["windy", "too much", "Y", "6.39"],
                                 ["good weather", "many", "N", "52.75"],
                                 ["rainy", "too much", "Y", "19.36"],
                                 ["windy", "heaps", "N", "18.39"],
                                 ["rainy", "too much", "Y", "17.32"],
                                 ["good weather", "too much", "Y", "18.29"],
                                 ["good weather", "some", "N", "79.21"],
                                 ["windy", "heaps", "Y", "29.63"],
                                 ["rainy", "heaps", "Y", "16.28"],
                                 ["rainy", "some", "N", "76.43"],
                                 ["good weather", "many", "N", "19.32"],
                                 ["windy", "heaps", "Y", "30.42"],
                                 ["good weather", "many", "N", "42.75"],
                                 ["rainy", "many", "N", "30.03"],
                                 ["rainy", "heaps", "Y", "32.69"],
                                 ["good weather", "many", "N", "19.99"],
                                 ["windy", "many", "N", "16.43"],
                                 ["rainy", "too much", "Y", "15.67"],
                                 ["good weather", "too much", "N", "30.40"]]),
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
            "good weather": 0,
            "rainy":1,
            "windy":2
        }
        return switcher.get(weatherState)

    def holidayTransmission(self, holidayState):
        switcher={
            "Y": 0,
            "N": 1
        }
        return switcher.get(holidayState)

    def fitModel(self):
        self.encodeData(self.X_train)
        self.clf = svm.SVC(kernel="linear", probability=True)
        self.clf.fit(self.X_train, self.y_train)

    def predict(self, dataset):
        self.encodeData(dataset)
        dataset = dataset.drop("visitorsOnThisDay")
        dataset = dataset.to_numpy().reshape(1, -1)
        prediction = pd.DataFrame(self.clf.classes_, columns=["value"])
        prediction["dist"] = self.clf.decision_function(dataset)[0]
        prediction = prediction.sort_values(by=["dist"], ascending=False)
        return prediction



