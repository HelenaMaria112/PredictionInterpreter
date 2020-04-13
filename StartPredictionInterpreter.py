from PredictionInterpreter import *
import pandas as pd
import numpy as np

if __name__ == '__main__':
    #define prediction function:
    def singlePredictJson(dataset):
        if dataset.iloc[1] == "good weather":
            return [{'predicted': "good weather", 'confidence': 0.9928, 'dist': 1.02},
                 {'predicted': "rainy", 'confidence': 0.28, 'dist': 1.785}]
        return [{'predicted': "rainy", 'confidence': 0.9908, 'dist': 0.55},
         {'predicted': "windy", 'confidence': 0.28, 'dist': 1.785}]

    #get your testData
    data = pd.DataFrame(np.array([["good weather", "94", "Y", "67.45"], ["rainy", "136", "N", "30.42"], ["windy","193", "Y", "98.25"]]),
                        columns=["weather", "visitorsPerDay", "holidayYN", "ticketPrice"])
    listOfNumericalColumns = ["ticketPrice"]
    standardColumns = data.columns.to_list()
    resultcolumn = "weather"
    _classes_ = data[resultcolumn].unique().tolist()

    #define predictor
    predictionInterpreter = PredictionInterpreterClass(singlePredictJson, listOfNumericalColumns, standardColumns, resultcolumn, _classes_, data)

    #call functions you want to use:
    #predictionInterpreter.data = data.sample(3)   # reduce data
    #predictionInterpreter.writeDistribution("visitorsPerDay")
    #predictionInterpreter.plotConfusionTable()
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
