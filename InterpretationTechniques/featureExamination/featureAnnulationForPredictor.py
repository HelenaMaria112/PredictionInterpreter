from Connections.predictor import *
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from InterpretationTechniques.PlotAndShow import *

def featureAnnulation(data, pr):
    '''
    :param data: pandas dataframe with datasets where each row represents a dataset
    :param resultColumnName: Name of column in data that contains actual results
    :param pr: Predictor of ML-System
    :return:
    '''
    resultColumnName = pr.resultColumn
    accuracies = pd.Series()
    predictions = pd.Series()
    origAccuracy = accuracy_score(data[resultColumnName], pr.predict(data), normalize=True)
    for column in data.columns.drop(resultColumnName):
        dataWOColumn = data.copy()
        dataWOColumn[column] = 0
        predictionColumn = pr.predict(dataWOColumn)
        accuracies[column] = accuracy_score(data[resultColumnName], predictionColumn, normalize=True, sample_weight=None)
        predictions[column] = predictionColumn
    accuracies = 1 - accuracies - (1 - origAccuracy)
    accuracies = accuracies.round(4)
    fig, ax = plt.subplots()
    rects = plt.bar(range(len(accuracies.index)), accuracies, data=accuracies.values)
    plt.xticks(range(len(accuracies.index)), accuracies.index)
    plt.xlabel('Parameters')
    plt.ylabel('Error rate')
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    save("featureImportance Annulation", plt=plt)