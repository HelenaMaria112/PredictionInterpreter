# https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from Connections.predictor import *

def plotCorrelationMatrix(data):
    X = data.drop("gknto", axis=1)  #independent columns
    y = data.loc[:,"gknto"]    #target column i.e price range
    #get correlations of each feature in dataset
    corrmat = X.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    #plot heat map
    g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")