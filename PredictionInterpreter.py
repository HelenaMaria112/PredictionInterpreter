# -*- encoding: utf-8 -*-
'''
Created on 04.10.2019

@author: areb
'''
from InterpretationTechniques.featureExamination.pdpForPredictor import *
from InterpretationTechniques.featureExamination.iceForPredictor import *
from InterpretationTechniques.basicData.confusionTableForPredictor import *
from InterpretationTechniques.featureExamination.pdpForPredictor import *
from InterpretationTechniques.featureExamination.shapForPredictor import *
from InterpretationTechniques.basicData.precisionForPredictor import *
from InterpretationTechniques.featureExamination.featureImportanceForPredictor import *
from InterpretationTechniques.featureExamination.CorrelationMatrixWithHeatmapForPredictor import *
from InterpretationTechniques.featureExamination.limeTextClassificatorForPredictor import *
from InterpretationTechniques.featureExamination.globalSurrogateModelForPredictor import *
from InterpretationTechniques.basicData.dataDistributionForPredictor import *
from InterpretationTechniques.featureExamination.featureAnnulationForPredictor import *
import pandas as pd
import sys

class MyPredictor(Predictor):
    '''
    Set all objects:
    resultColumn: Name of Column, in which actual results are saved (y_true), define as Sting e.g. 'animalTypes'
    standardColumns: all column Names of a dataset, define as array list of strings e.g. ['columnName1', 'columnName2', 'resultColumnName',...]
    _classes_: all type of classes that can appear in resultColumn, defined as in resultColumn e.g. np.array('cat', 'dog, 'mouse', 'elephant').astype(str)
    listOfNumericalColumns: if there are columns that are numerical, pass them as String Array e.g. ['bodySize', 'jumpingDistance',...]
    override singlePredictionJson-method
    '''
    def __init__(self, **args):
        '''
        make changes to passed args and change initialization, if needed
        :param args:
        '''
        super().__init__(**args)

    def __init__(self, singlePredictJson, listOfNumericalColumns, standardColumns, resultcolumn, _classes_, **args):
        '''
        make changes to passed args and change initialization, if needed
        :param singlePredictJson(dataset): method that predicts one dataset and returns results
                its parameter is one dataset from the data, that will be predicted. A dataset is one row of data
                its return is a list of results, where first entry is most likely result, second entry is second-most-likely result,...
                    each entry consists of class name, and at least one likelyhood-measure in this format:
                    [{'predicted': 'cat', 'confidence':0.9908, 'dist':3.785},{'predicted': 'dog', 'confidence':0.28, 'dist':1.785}]
                    the 'dist' needs to be set. The higher the distance is, the more secure the result is
        :param listOfNumericalColumns: for some Calculations it is necessary to know the columns that need to be interpreted numerically (e.g. age, costs, sallary, taxes ...)
        :param standardColumns: all columns in data, e.g. ['holidayYN', 'weatherCondition', 'visitorsADay', 'ticketPrice']
        :param resultcolumn: name of column in data that represents actual result, e.g. "visitorsADay"
        :param _classes_: all values in resultcolumn e.g. np.array([2345, 23423, 23423]).astype(Str) or np.array(["tenToInclFifty", "fiftyToInclNinety", "overNinety"])
        :param args:
        '''
        self.singlePredictJson = singlePredictJson
        self.listOfNumericalColumns = listOfNumericalColumns
        self.standardColumns = standardColumns
        self.resultColumn = resultcolumn
        self._classes_ = _classes_
        super().__init__(**args)

    def singlePredictJson(self, dataset):
        '''
        predicts one dataset and returns results
        :param dataset: dataset from data, that will be predicted. A dataset is one row of data
        :return:    list of results, where first entry is most likely result, second entry is second-most-likely result,...
                    each entry consists of class name, and at least one likelyhood-measure in this format:
                    [{'predicted': 'cat', 'confidence':0.9908, 'dist':3.785},{'predicted': 'dog', 'confidence':0.28, 'dist':1.785}]
                    the 'dist' needs to be set. The higher the distance is, the more secure the result is
        '''

class PredictionInterpreterClass:
    def __init__(self, singlePredictJson, listOfNumericalColumns, standardColumns, resultcolumn, _classes_, data):
        self.singlePredictJson=singlePredictJson
        self.listOfNumericalColumns=listOfNumericalColumns
        self.standardColumns=standardColumns
        self.resultcolumn=resultcolumn
        self._classes_=_classes_
        self.data=data

    def initPredictor(self):
        self.myPredictor = MyPredictor(self.singlePredictJson, self.listOfNumericalColumns, self.standardColumns, self.resultcolumn, self._classes_)

    def writeDistribution(self, groupbyColumn):
        self.initPredictor()
        writeDistribution(self.data, groupbyColumn)
    def plotConfusionTable(self):
        self.initPredictor()
        plotConfusionTable(self.data, self.myPredictor)
    def printImportanceEli5(self, distanceAnalysis = True, exceptedColumns = None):
        self.initPredictor()
        printImportanceEli5(self.data, self.myPredictor, distanceAnalysis=distanceAnalysis, exceptedColumns = exceptedColumns)
    def featureAnnulation(self):
        self.initPredictor()
        featureAnnulation(self.data, self.myPredictor)
    def plotIce(self):
        self.initPredictor()
        plotIce(self.data, self.myPredictor.setReturnDistanceOfClass(True))
    def plotpdpOfDistanceToTrueResultPdpbox(self, featureToExamine = None, featuresToExamine = None, exceptedColumns= None):
        '''
        :param featureToExamine: a single feature to examine, e.g. "weather"
        :param featuresToExamine: touples of features to examine, e.g. ["weather", "holidayYN"]
        :param exceptedColumns: columns that are not examined, e.g. "ticketPrice"
        :return: plot
        '''
        self.initPredictor()
        plotpdpOfDistanceToTrueResultPdpbox(self.data, self.myPredictor.resultColumn, featureToExamine = featureToExamine, featuresToExamine = featuresToExamine, exceptedColumns=exceptedColumns,  pr = self.myPredictor.setReturnDistanceOfClass(True))

    def plotpdpOfDistanceToTrueResultSklearn(self, listOfColumnsToExamine = None):
        '''
        :param listOfColumnsToExamine: list of all columns to examine e.g. ["weather", "holidayYN", "ticketPrice"]
        :return:
        '''
        self.initPredictor()
        if listOfColumnsToExamine == None:
            listOfColumnsToExamine = self.myPredictor.standardColumnsNoResultColumn()
        plotpdpOfDistanceToTrueResultSklearn(self.data, listOfColumnsToExamine, pr = self.myPredictor.setReturnDistanceOfClass(True))

    def plotpdpOfDistanceToTrueResultSklearn2D(self):
        self.initPredictor()
        plotpdpOfDistanceToTrueResultSklearn2D(self.data, pr = self.myPredictor.setReturnDistanceOfClass(True))

    def globalSurrogateModel(self):
        self.initPredictor()
        self.myPredictor.callingFunction = "globalSurrogate"
        globalSurrogateModel(self.data, pr = self.myPredictor)  # use enough datasets for this technique

# the following techniques need jupyter notebook to be run. --------------------------
def limeTextClassificationJupyter(dataset, singlePredictJson, listOfNumericalColumns, standardColumns,
                                                       resultcolumn, _classes_, data):  # function to be called from Jupyter notebook
    print("Start predictions")
    predictionInterpreter = PredictionInterpreterClass(singlePredictJson, listOfNumericalColumns, standardColumns,
                                                       resultcolumn, _classes_, data)
    pr = predictionInterpreter.initPredictor()
    pr.callingFunction = "TextClassifier"
    return limeTextClassification(dataset, data, pr=pr)

def limeTextClassificationJupyter(singlePredictJson, listOfNumericalColumns, standardColumns,
                                                       resultcolumn, _classes_, data):  # function to be called from Jupyter notebook
    predictionInterpreter = PredictionInterpreterClass(singlePredictJson, listOfNumericalColumns, standardColumns,
                                                       resultcolumn, _classes_, data)
    pr = predictionInterpreter.initPredictor()
    pr.setReturnDistanceOfClass(True)
    pr.callingFunction = "shap"
    print("Start predictions")
    return plotShap(data, columnsThatAreRoundedToTwoDecimals = pr.listOfNumericalColumns, pr = pr)