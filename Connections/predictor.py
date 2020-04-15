'''
Created on 04.10.2019

@author: areb
'''
import threading
import requests  # introduction see https://realpython.com/python-requests/
import pandas as pd
import json
import numpy as np
import warnings
import time
import itertools

from pygments.lexer import default

class Predictor:
    fit = None
    _estimator_type = "classifier"
    dataValuesEncoded = False
    counter = 0
    callingFunction = None
    standardColumns = None
    _classes_ = None
    resultColumn = ""
    listOfNumericalColumns = None

    def __init__(self, standardColumns = None, resultColumn = None, _classes_ = None, dataset=None, returnDistanceOfClass=False, callingFunction = None, listOfNumericalColumns = None, decodedColumns = None):
        self.lock = threading.Lock()
        if listOfNumericalColumns is not None:
            self.listOfNumericalColumns = listOfNumericalColumns
        if standardColumns  is not None:
            self.standardColumns = standardColumns
        if callingFunction is not None:
            self.callingFunction=callingFunction
        if returnDistanceOfClass is not None:
            self.returnDistanceOfClass = returnDistanceOfClass
            if (returnDistanceOfClass):
                self._estimator_type = "regressor"
        if dataset is not None:
            self.dataset = dataset  #needed for TextExplainer only
        if resultColumn is not None:
            self.resultColumn = resultColumn
        if _classes_ is not None:
            self._classes_ = _classes_
            self._classes_000 = np.append(self._classes_, np.array(['000', '001']))

    def getResultColumn(self):
        return self.resultColumn

    def setReturnDistanceOfClass(self, returnDistanceOfClass):
        self.returnDistanceOfClass = returnDistanceOfClass
        if(returnDistanceOfClass):
            self._estimator_type = "regressor"
        else:
            self._estimator_type = "classifier"
        return self

    def encode(self, data, exceptedColumns = None):
        self.standardColumns = data.columns
        if not isinstance(data, pd.DataFrame):
            raise ValueError('data must be pandas dataframe')

        self.dataValuesEncoded = True

        self.decodedColumns= data.columns
        if exceptedColumns:
            self.decodedColumns=self.decodedColumns.drop(exceptedColumns)
        data[self.listOfNumericalColumns] = data[self.listOfNumericalColumns].astype(float).astype(str)

        self.createEncodingDictionary(data)
        return self.encodeData(data)

    def createEncodingDictionary(self, data):
        self.encodingDictionary = pd.Series()
        for column in self.decodedColumns:
            if self.listOfNumericalColumns is not None and column in self.listOfNumericalColumns:
                self.encodingDictionary[column] = data.loc[:][column].astype(float).sort_values().astype(str).unique()
            else:
                self.encodingDictionary[column] = data.loc[:][column].sort_values().unique()

    def encodeData(self, data):
        encodedData = pd.DataFrame(index=data.index, columns=data.columns)
        for rowIndex in range(len(encodedData)):
            for column in encodedData.columns:
                realValue = data.iloc[rowIndex].loc[column]
                if column not in self.decodedColumns:
                    encodedData.iloc[rowIndex].loc[column] = realValue
                else:
                    encodedData.iloc[rowIndex].loc[column] = np.where(self.encodingDictionary[column] == realValue)[0][0]
        return encodedData

    def decode(self, data):
        if self.counter ==1:
            print(data.index)
        print(data.values)
        if(isinstance(data, pd.DataFrame)):
            decodedData = pd.DataFrame(index=data.index, columns = data.columns)
            for rowIndex, row in data.iterrows():
                for column in data.columns:
                    code = data.iloc[rowIndex].loc[column]
                    if column in self.decodedColumns:
                        decodedData.iloc[rowIndex].loc[column] = code
                    else:
                        decodedData.iloc[rowIndex].loc[column]=self.encodingDictionary[column][int(code)]
            return decodedData
        if(isinstance(data, pd.Series)):
            decodedData = pd.Series(index=data.index)
            for index in data.index:
                if index in self.decodedColumns:
                    decodedData[index]=self.encodingDictionary[index][int(data[index])]
                else:
                    decodedData[index]= data[index]
            return decodedData


    def predict(self, data, predict_kwds=None, valueConfidenceDistance = "value", distanceClass = None):
        if self.callingFunction == "shap":
            data[self.resultColumn] = data.index
        if distanceClass is not None:
            self.distanceClass = distanceClass
        if(isinstance(data,np.ndarray)):
            if np.size(data,1) == len(self.standardColumns):
                data = pd.DataFrame(data=data, columns=self.standardColumns)
            else:
                data = pd.DataFrame(data=data, columns=self.standardColumnsNoResultColumn())

        if (data.ndim == 1):
            return self.predictSingle(data, valueConfidenceDistance)

        if valueConfidenceDistance == 'value' and self.returnDistanceOfClass == False:
            predictions = np.array(data[self.resultColumn])
            predictions.fill('NA')
            for rowIndex in range(len(data)):
                predictions[rowIndex] = self.predictSingle(data.iloc[rowIndex], valueConfidenceDistance)
            return predictions

        predictions = np.zeros(len(data))
        for rowIndex in range(len(data)):
            predictions[rowIndex] = self.predictSingle(data.iloc[rowIndex], valueConfidenceDistance)
        return predictions

    def standardColumnsNoResultColumn(self):
        tempColumns = list(self.standardColumns)
        tempColumns.remove(self.resultColumn)
        return tempColumns

    def predictSingle(self, dataset, valueConfidenceDistance):
        if self.dataValuesEncoded:
            print(dataset.values)
            dataset = self.decode(dataset)

        jsonObject = self.singlePredictJson(dataset)
        if self.returnDistanceOfClass:
            if not isinstance(dataset.loc[self.resultColumn], str):
                distanceClass = dataset.loc[self.resultColumn].astype(str)
            else:
                distanceClass = dataset.loc[self.resultColumn]
            for subPrediction in jsonObject:
                if subPrediction["predicted"] == distanceClass:
                    print(str(self.counter) + "  " + distanceClass + "  ->  " + str(max(subPrediction["dist"], 0.0)))
                    return max(subPrediction["dist"], 0.0)
            #print("distance for true value not found")
            print(str(self.counter) + str(distanceClass) + "  -> 0.0 ")
            return 0.0

        mostLikelyGuess = jsonObject[0]
        if valueConfidenceDistance == "value":
            print(str(mostLikelyGuess['predicted']) + "   " + str(self.counter))
            return mostLikelyGuess['predicted']
        elif valueConfidenceDistance == "confidence":
            return mostLikelyGuess['confidence']
        elif valueConfidenceDistance == "distance":
            return mostLikelyGuess['dist']
        elif valueConfidenceDistance == "valueDistance":
            return [mostLikelyGuess['predicted'], mostLikelyGuess['dist']]

    def predict_proba_ProbabilityMatrix(self, data):
        '''
        :param data:
        :return:
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        '''
        return self.predict_proba(data, "probability")

    def predict_proba(self, data):
        """
        :param data:
        :return:
        predictionMatrix: array-like, shape (n_samples, n_classes)
                          Returns the probability of the sample for each class in
                          the model.
        """
        if self._classes_ is None:
            raise ValueError('No dataClasses specified in predictor declaration')

        if self.callingFunction == "TextClassifier":
            return self.predict_probaLime(data)

        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data = data, columns = self.standardColumns)
        if isinstance(data, list):
            data = pd.DataFrame(data)

        predictionMatrix = pd.DataFrame(index=data.index, columns=self._classes_).fillna(0.0)

        for rowIndex, dataset in data.iterrows():
            predictionResults = self.singlePredictJson(dataset)
            if self.callingFunction != "shap":
                predictionMatrix = self.setAllDistancesInComparison(predictionMatrix, predictionResults, rowIndex)
            else:
                predictionMatrix = self.setTopPredictionToOne(predictionMatrix, predictionResults, rowIndex)
        if self.callingFunction == "shap":
            predictionMatrix = predictionMatrix.replace(0.0, 0.01)
            print(predictionMatrix)
            return predictionMatrix
        if self.callingFunction == "TextClassifier":
            print(predictionMatrix.to_numpy())
            predictionMatrix.to_numpy()
        else:
            print(predictionMatrix)
            return predictionMatrix

    def predict_probaLime(self, dataStrings):
        '''
        dt = np.dtype({'names': self.dataClasses,
                        'formats': ['str']*len(self.dataClasses) })

        predictionMatrix = np.zeros((len(dataStrings),len(self.dataClasses)), names = self.dataClasses)
        '''
        data = np.zeros((len(dataStrings), len(self._classes_000)))
        predictionMatrix = pd.DataFrame(data, index = dataStrings, columns = self._classes_000)

        for dataStringIndex in range(len(dataStrings)):
            predictionResults = self.singlePredictJson(dataStrings[dataStringIndex])
            for result in predictionResults:
                result['dist'] = max(result['dist'], 0.0)
            DistSum = sum(singleResult['dist'] for singleResult in predictionResults)
            if len(predictionResults) == 1:
                result = predictionResults[0]
                predictionMatrix.iloc[dataStringIndex].loc[(result['predicted'])] = max(min(result['dist'], 0.99),0.01)
                predictionMatrix.iloc[dataStringIndex].iloc[-1] = 1-predictionMatrix.iloc[dataStringIndex].loc[(result['predicted'])]
            elif len(predictionResults) == 0:
                predictionMatrix.iloc[dataStringIndex][-1] = 0.5
                predictionResults.iloc[dataStringIndex][-2] = 0.5
            else:
                for result in predictionResults:
                    predictionMatrix.iloc[dataStringIndex].loc[(result['predicted'])] =  (result['dist']/DistSum).__round__(5)
                percentualPredictionsSumWithoutFirst = sum(predictionMatrix.iloc[dataStringIndex]) - predictionMatrix.iloc[dataStringIndex].loc[(predictionResults[0]['predicted'])]
                predictionMatrix.iloc[dataStringIndex].loc[(predictionResults[0]['predicted'])] =  1.0 - percentualPredictionsSumWithoutFirst

            if sum(predictionMatrix.iloc[dataStringIndex]) != 1.0 or max(predictionMatrix.iloc[dataStringIndex]) == 1.0:
                print("Row not 1.0 : " + str(dataStringIndex))
                predictionMatrix.iloc[dataStringIndex].values
                sum(predictionMatrix.iloc[dataStringIndex])

        return predictionMatrix.to_numpy().astype(np.float64)

    def setTopPredictionToOne(self, predictionMatrix, predictionResults, rowIndex):
        predictionMatrix.iloc[rowIndex].loc[(predictionResults[0]['predicted'])] = 1
        return predictionMatrix

    def setAllDistancesInComparison(self, predictionMatrix, predictionResults, rowIndex):
        distanceSum = sum(singleResult['dist'] for singleResult in predictionResults)
        # print(min(predictionResults[0], 11111111111111111111111111))
        if distanceSum != 0.0:
            for result in predictionResults:
                predictionMatrix.iloc[rowIndex].loc[(result['predicted'])] = result['dist'] / distanceSum
            percentualPredictionsSumWithoutFirst = sum(predictionMatrix.iloc[rowIndex]) - \
                                                   predictionMatrix.iloc[rowIndex].loc[
                                                       predictionResults[0]['predicted']]
            predictionMatrix.iloc[rowIndex].loc[
                (predictionResults[0]['predicted'])] = 1 - percentualPredictionsSumWithoutFirst
        return predictionMatrix

    def setAllDistances(self, predictionMatrix, predictionResults, rowIndex):
        distanceSum = sum(singleResult['dist'] for singleResult in predictionResults)
        # print(min(predictionResults[0], 11111111111111111111111111))
        if distanceSum != 0.0:
            for result in predictionResults:
                predictionMatrix.iloc[rowIndex].loc[(result['predicted'])] = max(result['dist'], 0.0)
        return predictionMatrix


    def predict_proba_ErrorMatrix(self, data):
        '''
        :param data:
        :return:
        pandas matrix of shape = [n_samples, n_classes]:
        The class errors of the input samples
        '''
        if self._classes_ is None:
            raise ValueError('No dataClasses specified in predictor declaration')

        predictionMatrix = pd.DataFrame(index=data.index, columns=self._classes_).fillna(0.0)
        for rowIndex in range(len(data)):
            predictionResults = self.singlePredictJson(data.iloc[rowIndex])
            distanceSum = sum(singleResult['dist'] for singleResult in predictionResults)
            if distanceSum != 0.0:
                for result in predictionResults:
                    predictionMatrix.iloc[rowIndex].loc[(result['predicted'])] = result['dist'] / distanceSum
                percentualPredictionsSumWithoutFirst = sum(predictionMatrix.iloc[rowIndex]) - predictionMatrix.iloc[rowIndex].loc[predictionResults[0]['predicted']]
                predictionMatrix.iloc[rowIndex].loc[(predictionResults[0]['predicted'])] = 1 - percentualPredictionsSumWithoutFirst
        return predictionMatrix

    def singlePredictJson(self, dataset):
        #corpus to add singlePredict. this method is overwritten in PredictionInterpreter.py
        response = None
        return response.json()

    def unsortedColumnCombinations(self, data, trueResultColumn):
        columnList = data.columns.to_list()
        columnList.remove(trueResultColumn)
        return columnList + list(itertools.combinations(columnList, 2))  # list with all column combinatins


