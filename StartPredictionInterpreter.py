from PredictionInterpreter import *
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from DummyMLModel import *


def getTestData():
    return pd.DataFrame(np.array([["good weather", "many", "Y", "67.45"],
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

if __name__ == '__main__':
    #get data and object that has singlepredict
    dm = DummyMLModel()
    data = dm.testdata

    #define necessary variables for techniques
    listOfNumericalColumns = ["ticketPrice"]
    standardColumns = data.columns.to_list()
    resultcolumn = "visitorsOnThisDay"
    _classes_ = data[resultcolumn].unique().tolist()

    #define predictor
    predictionInterpreter = PredictionInterpreterClass(dm.predict, listOfNumericalColumns, standardColumns, resultcolumn, _classes_, data)
    #call functions you want to use:
    #predictionInterpreter.writeDistribution("visitorsOnThisDay")
    predictionInterpreter.plotConfusionTable()
    #predictionInterpreter.printImportanceEli5(exceptedColumns = "visitorsPerDay")
    #predictionInterpreter.printImportanceEli5(distanceAnalysis=True)
    #predictionInterpreter.featureAnnulation()
    #predictionInterpreter.plotIce()
    #predictionInterpreter.plotpdpOfDistanceToTrueResultPdpbox(featureToExamine="ticketPrice")
    #predictionInterpreter.plotpdpOfDistanceToTrueResultPdpbox(featuresToExamine=["holidayYN", "ticketPrice"])
    #predictionInterpreter.plotpdpOfDistanceToTrueResultPdpbox(featureToExamine="ticketPrice", featuresToExamine=["holidayYN", "ticketPrice"])
    #predictionInterpreter.plotpdpOfDistanceToTrueResultSklearn()
    #predictionInterpreter.plotpdpOfDistanceToTrueResultSklearn2D()
    #predictionInterpreter.globalSurrogateModel()
    print("finished")
