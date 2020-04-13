from Connections.predictor import *
import numpy as np
from sklearn.metrics import accuracy_score #define a score function. In this case I use accuracy
from eli5.permutation_importance import get_score_importances  # This function takes only numpy arrays as inputs
from matplotlib import pyplot as plt
from InterpretationTechniques.PlotAndShow import *

#see documentation at https://towardsdatascience.com/how-to-find-feature-importances-for-blackbox-models-c418b694659d
class printImportanceEli5:

    def score(self, x, y):
        #score rewards prediction accuracy. Here it is between 0 and 1.
        if self.distanceAnalysis:
            #score system by probability of prediction (e.g. distance to hyperline in svm)
            return self.pr.predict(x).mean()

        #score system by accuracy
        y_pred = self.pr.predict(x).astype(int).astype(str)
        print("-------")
        return accuracy_score(y.astype(str), y_pred)

    def __init__(self, data, pr, distanceAnalysis=False, exceptedColumns = None ):
        '''
        :param data: pandas dataframe with datasets where each row represents a dataset
        :param resultColumnName: Name of column in data that contains actual results
        :param pr: Predictor of ML-System
        :param distanceAnalysis: if set to true, distances are used as measurement for correctness of result
        plots and saves feature importance plot by using ELI5 and Accuracy
        '''
        resultColumnName = pr.resultColumn
        self.pr = pr
        self.distanceAnalysis = distanceAnalysis
        data = self.pr.encode(data, exceptedColumns = exceptedColumns)
        X = data
        y = data[resultColumnName]  # target column i.e price range apply SelectKBest class to extract top 10 best features
        if distanceAnalysis:
            self.pr.returnDistanceOfClass = True
        else:
            X = X.drop([resultColumnName], axis=1) #independent columns.

        base_score, score_decreases = get_score_importances(self.score, np.array(X), y)
        feature_importances = np.mean(score_decreases, axis=0)

        feature_importance_dict = {}
        for i, feature_name in enumerate(X.columns):
            feature_importance_dict[feature_name] = feature_importances[i]
        print(dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:4]))
        self.f_importances(feature_importance_dict, resultColumnName)

    def f_importances(self, feature_importance_dict, resultColumnName):
        '''  nice way of plotting feature importances as seen in https://stackoverflow.com/questions/41592661/determining-the-most-contributing-features-for-svm-classifier-in-sklearn
        :param feature_importance_dict:
        :param resultColumnName:
        saves plot with feature Importance
        '''
        imp = feature_importance_dict.values()
        names = feature_importance_dict.keys()
        imp,names = zip(*sorted(zip(imp,names)))
        if self.distanceAnalysis:
            posGknto=names.index(resultColumnName)
            imp=imp[:posGknto]+imp[posGknto+1:]
            names=names[:posGknto]+names[posGknto+1:]

        plt.barh(range(len(names)), imp, align='center')
        plt.yticks(range(len(names)), names)

        save("featureImportance ", plt=plt)
