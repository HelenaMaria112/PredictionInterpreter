from PredictionInterpreter import *
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from DummyMLModel import *

if __name__ == '__main__':
    #get data and object that has singlepredict in correct format
    dm = DummyMLModel()
    data = dm.testdata

    #define necessary variables for techniques
    listOfNumericalColumns = ["ticketPrice"]
    standardColumns = data.columns.to_list()
    resultcolumn = "visitorsOnThisDay"
    _classes_ = data[resultcolumn].unique().tolist()
    resultIsContinuous = False

    #define predictor
    predictionInterpreter = PredictionInterpreterClass(dm.predict, listOfNumericalColumns, standardColumns, resultcolumn, _classes_, data, resultIsContinuous)
    #call functions you want to use:
    predictionInterpreter.plotpdpOfDistanceToTrueResultSklearn() # only works if called without any prior methods
    predictionInterpreter.plotpdpOfDistanceToTrueResultSklearn2D()
    predictionInterpreter.writeDistribution("visitorsOnThisDay")
    predictionInterpreter.plotConfusionTable()
    predictionInterpreter.printImportanceEli5(exceptedColumns = resultcolumn)
    predictionInterpreter.printImportanceEli5(distanceAnalysis=True)
    predictionInterpreter.featureAnnulation(annulationValue = "0")
    predictionInterpreter.plotIce()
    predictionInterpreter.plotpdpOfDistanceToTrueResultPdpbox(featureToExamine="ticketPrice")
    predictionInterpreter.plotpdpOfDistanceToTrueResultPdpbox(featuresToExamine=["holidayYN", "ticketPrice"])
    predictionInterpreter.plotpdpOfDistanceToTrueResultPdpbox(featureToExamine="ticketPrice", featuresToExamine=["holidayYN", "ticketPrice"])
    predictionInterpreter.globalSurrogateModel()
    print("finished")

