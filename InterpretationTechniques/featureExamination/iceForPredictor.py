'''
Created on 04.10.2019

@author: areb
'''
import pycebox.ice as pIce
from Connections.predictor import *
import matplotlib.pyplot as plt
from InterpretationTechniques.PlotAndShow import *

# https://github.com/AustinRochford/PyCEbox/blob/master/pycebox/ice.py
def plotIce(data, pr):
    '''
    :param data: pandas dataframe with datasets where each row represents a dataset
    :param resultColumnName: Name of column in data that contains actual results
    :param pr: Predictor of ML-System
    saves and plots ICE
    '''
    pr.setReturnDistanceOfClass(True)
    resultColumnName = pr.resultColumn
    for i in pr.listOfNumericalColumns:
        data[i]= data[i].astype(float).round(2).astype(str)
    data = pr.encode(data)

    columnCombinations = pr.unsortedColumnCombinations(data, resultColumnName)
    for columnCombination in columnCombinations:
        if not isinstance(columnCombination, tuple):
            iceResult = pIce.ice(data, columnCombination, pr.predict, num_grid_points=None)

            ax = pIce.ice_plot(iceResult, frac_to_plot=1.,
                              plot_points=True, point_kwargs=None,
                              x_quantile=False, plot_pdp=True,
                              centered=False, centered_quantile=0.,
                              color_by=None, cmap=None,
                              ax=None, pdp_kwargs=None)
            ax.set_ylabel("Distance to Hyperplane of true result")
            ax.set_xlabel(columnCombination)
            ax.set_title("ICE for " + columnCombination)
            lines = ax.lines
            for lineIndex in range(len(lines)):
                lines[lineIndex].set_label("Dataset "+str(lineIndex))
            lines[len(lines)-1].set_label("Pdp")
            #ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

            for line in ax.lines:
                line.set_color("k")
                line._linewidth = 0.5
            lines[-1].linewidth=1
            lines[-1].set_color("r")
            xValues = pr.encodingDictionary[columnCombination]
            ax.set_xticks(np.arange(1, len(xValues), 1))
            ax.set_xticklabels(xValues[1:])
            ax.tick_params(axis='both', which='major', labelsize=6)
            ax.tick_params(axis='both', which='minor', labelsize=6)
            plt.xticks(rotation=90)
            saveName="ice"+str(columnCombination)
            save(saveName, plt=plt)

def bspSammlung(data):
    import pandas as pd
    pr= Predictor(returnDistanceOfClass=True)
    d = data.sample(3)
    y = pd.DataFrame(index=range(9), columns=d.columns)
    for i in range(len(d)):
        y.iloc[3 * i + 0] = d.iloc[i]
        y.iloc[3 * i + 0].loc["brutto"] = d.iloc[0].loc["brutto"]
        y.iloc[3 * i + 1] = d.iloc[i]
        y.iloc[3 * i + 1].loc["brutto"] = d.iloc[1].loc["brutto"]
        y.iloc[3 * i + 2] = d.iloc[i]
        y.iloc[3 * i + 2].loc["brutto"] = d.iloc[2].loc["brutto"]
    y.loc[:]["gknto"] = pr.predict(y).astype(str)
    ystr = ""
    for r in range(len(y)):
        for c in range(len(y.columns)):
            ystr = ystr + y.iloc[r, c][0:min(50, len(y.iloc[r, c]))] + "  \n"
        ystr = ystr + "\n"
    f = open("bspDaten.txt", "w+")
    f.write(ystr)
    f.close()

    import pandas as pd
    d = data.sample(3)
    y = pd.DataFrame(index=range(9), columns=d.columns)
    for i in range(len(d)):
        y.iloc[3 * i + 0] = d.iloc[i]
        y.iloc[3 * i + 0].loc["text"] = d.iloc[0].loc["text"]
        y.iloc[3 * i + 1] = d.iloc[i]
        y.iloc[3 * i + 1].loc["text"] = d.iloc[1].loc["text"]
        y.iloc[3 * i + 2] = d.iloc[i]
        y.iloc[3 * i + 2].loc["text"] = d.iloc[2].loc["text"]
    y.loc[:]["gknto"] = pr.predict(y).astype(str)
    ystr = ""
    for r in range(len(y)):
        for c in range(len(y.columns)):
            ystr = ystr + y.iloc[r, c][0:min(50, len(y.iloc[r, c]))] + "  \n"
        ystr = ystr + "\n"
    f = open("bspDaten.txt", "w+")
    f.write(ystr)
    f.close()

def writedataToFileWPredictionhead3(data, pr):
    d=data[:3]
    y = pd.DataFrame(index=range(9), columns=(d.columns))
    for i in range(len(d)):
        y.iloc[3 * i + 0] = d.iloc[i]
        y.iloc[3 * i + 0].loc["text"] = d.iloc[0].loc["text"]
        y.iloc[3 * i + 1] = d.iloc[i]
        y.iloc[3 * i + 1].loc["text"] = d.iloc[1].loc["text"]
        y.iloc[3 * i + 2] = d.iloc[i]
        y.iloc[3 * i + 2].loc["text"] = d.iloc[2].loc["text"]

    y["Result"] = pr.predict(y).astype(str)
    ystr = ""
    for r in range(len(y)):
        for c in range(len(y.columns)):
            ystr = ystr + y.iloc[r, c][:min(10, len(y.iloc[r, c])-1)] + "  \n"
        ystr = ystr + "\n"
    f = open("bspDaten.txt", "w+")
    f.write(ystr)
    f.close()