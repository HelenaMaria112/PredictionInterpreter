from sklearn.inspection import plot_partial_dependence
from pdpbox import pdp, info_plots
from InterpretationTechniques.PlotAndShow import *
from matplotlib.ticker import MaxNLocator
import threading
import datetime
from Connections.data import saveDataAnalysis

#after https://github.com/SauceCat/PDPbox/blob/master/tutorials/pdpbox_regression.ipynb
class Locker:
    lock = threading.Lock()

def targetDistribution(data, featureToExamine, resultColumnName):
    fig, axes, summary_df = info_plots.target_plot(
        df=data, feature=featureToExamine, feature_name=featureToExamine, target=resultColumnName
    )
    save("targetDistribution", fig=fig, plt=plt)

def predictionDistribution(data, pr, featureToExamine):
    fig, axes, summary_df = info_plots.actual_plot(
    model=pr, X=data, feature=featureToExamine, feature_name=featureToExamine, predict_kwds={})
    save("predictionDistribution", fig=fig, plt=plt)

def pdpPdpbox(data, pr, featureToExamine):
    pdpValues = pdp.pdp_isolate(model=pr, dataset=data, model_features=data.columns, feature=featureToExamine)
    figPdp, axesPdp = pdp.pdp_plot(pdpValues, featureToExamine, plot_lines=True, frac_to_plot=min(100,len(data)))
    for line in axesPdp["pdp_ax"].lines:
        line._alpha = 1
    save("pdpPdpboxIsolate", plt=plt)

    fig, axes = pdp.pdp_plot(
        pdpValues, featureToExamine, plot_lines=True, frac_to_plot=min(100,len(data)), x_quantile=True,
        plot_pts_dist=True, show_percentile=True
    )
    for line in axes["pdp_ax"]["_pdp_ax"].lines:
        line._alpha = 1
    for line in axes["pdp_ax"]["_count_ax"].lines:
        line._alpha = 1
    save("pdpPdpboxPlot", plt=plt, fig=fig)

def targetDistributionNumericFeature(data,featureToExamine, resultColumnName, show_percentile = True):
    fig, axes, summary_df = info_plots.target_plot(
        df=data, feature=featureToExamine, feature_name=featureToExamine, target=resultColumnName, show_percentile=show_percentile
    )
    save("targetDistributionNumericFeature", plt=plt, fig=fig)

def interactionPlotReal(data, featuresToExamine, pr):
    fig, axes, summary_df = info_plots.target_plot_interact(
        df=data, features=featuresToExamine,
        feature_names=featuresToExamine, target=pr.resultColumn
    )
    save("interactionPlotReal", plt=plt, fig=fig)


def interactionPlotPredicted(data, pr, featuresToExamine):
    fig, axes, summary_df = info_plots.actual_plot_interact(
    model=pr, X=data,
    features=featuresToExamine,
    feature_names=featuresToExamine
    )
    save("interactionPlotPredicted", plt=plt, fig=fig)

def plotpdpOfDistanceToTrueResultPdpbox(data, resultColumnName, pr, featureToExamine = None, featuresToExamine = None, exceptedColumns=None, resultIsContinuous = True):
    '''
    :param data: pandas dataframe with datasets where each row represents a dataset
    :param resultColumnName: Name of column in data that contains actual results
    :param pr: Predictor of ML-System
    :param featureToExamine: features of which PDP will be calculated
    saves and plots one-dimensional PDPs that are calculated with PDPbox
    '''
    pr.setReturnDistanceOfClass(True)
    resultColumnName = pr.resultColumn
    data=pr.encode(data, exceptedColumns=exceptedColumns)
    if featureToExamine is not None:
        if resultIsContinuous == True:
            targetDistribution(data, featureToExamine, resultColumnName)
            targetDistributionNumericFeature(data, featureToExamine,
                                             resultColumnName)  # very similar to normal distribution
        predictionDistribution(data, pr, featureToExamine)
        pdpPdpbox(data, pr, featureToExamine)
    if featuresToExamine is not None:
        if resultIsContinuous == True:
            interactionPlotReal(data, featuresToExamine, pr)
        interactionPlotPredicted(data, pr, featuresToExamine)

def plotpdpOfDistanceToTrueResultSklearn(data, subplots, pr ):
    '''
    :param data: pandas dataframe with datasets where each row represents a dataset
    :param subplots: indicates columns to examine in pdp plot
    :param pr: Predictor of ML-System
    saves and plots indicated PDPplots that are calculated with sklearn
    '''
    pr.setReturnDistanceOfClass(True)
    resultColumnName = pr.resultColumn
    data = pr.encode(data)

    pr.standardColumnsNoResultColumn()
    plot_partial_dependence(pr, data, subplots, feature_names=pr.standardColumns)

    for i in range(len(subplots)):
        ax = plt.gcf().axes[i]
        spreadfourSubplotsHorizontally(ax, i)
        subplotXLabel = subplots[i] #ax.get_xlabel()
        ticks(ax, pr, subplotXLabel, "x")
        plt.title("PDP for "+ subplotXLabel)
    plt.gcf().set_size_inches(30,7)

    save("plot_partial_dependence textBruttoClient", plt=plt)
    writeDictToFile(pr.encodingDictionary, pr.decodedColumns)

def spreadfourSubplotsHorizontally(ax, i):
    plotwidth = 0.15
    space = 0.07
    posX0 = space + space * i + plotwidth * i
    ax.set_position([posX0, 0.2, plotwidth, 0.7])
    print(ax.get_position())

def ticks(ax, pr, subplotlabel, axesToLabel):
    values = pr.encodingDictionary[subplotlabel]
    for x in range(len(values)):
        values[x] = values[x][:(min(15, len(values[x])))]
    if subplotlabel == "brutto":
        values = values.astype(float).round(2)
    if "x" in axesToLabel:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticklabels(values)
        plt.sca(ax)
        plt.xticks(rotation=90)
    if "y" in axesToLabel:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_yticklabels(values)
    plt.margins(0.02) # set space inbetween ticks

def plotpdpOfDistanceToTrueResultSklearn2D(data, pr):
    '''
    :param data: pandas dataframe with datasets where each row represents a dataset
    :param resultColumnName: Name of column in data that contains actual results
    :param pr: Predictor of ML-System
    saves and plots two-dimensional PDPplots that are calculated with sklearn
    '''
    pr.setReturnDistanceOfClass(True)
    resultColumnName = pr.resultColumn
    saveDataAnalysis(data)
    data = pr.encode(data)
    writeDictToFile(pr.encodingDictionary, pr.decodedColumns)
    columnCombinations = pr.unsortedColumnCombinations(data, resultColumnName)
    for columnCombination in columnCombinations:
        if type(columnCombination) == tuple:
            plot_partial_dependence(pr, data, [columnCombination], feature_names=pr.standardColumns)
            ticks(plt.gca(), pr, columnCombination[0], "x" )
            ticks(plt.gca(), pr, columnCombination[1], "y" )
            plt.title("PDP for " + columnCombination[0] + " and " + columnCombination[1])
            save("pdp"+str(columnCombination), plt=plt)


def writeDictToFile(encodingDictionary, decodedColumns):
    ystr=""
    for column in decodedColumns:
        ystr= ystr + column + ": \n"
        for elementIndex in range(len(encodingDictionary[column])):
            ystr = ystr + encodingDictionary[column][elementIndex] + "\n"
            if elementIndex % 10 == 0 and elementIndex !=0:
                ystr = ystr + "\n"
        ystr = ystr + "\n \n"
    docName="EncodingDictionary.txt" + str(datetime.datetime.now())[:19].replace(":","-") + ".txt"
    f = open(docName, "w+")
    f.write(ystr)
    f.close()
