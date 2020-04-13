from distutils.command.config import config
from Connections.predictor import *
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from InterpretationTechniques.PlotAndShow import *

def plotConfusionTable(data, pr):
    '''
    :param data: pandas dataframe with datasets where each row represents a dataset
    :param resultColumnName: Name of result column in data
    :param pr: predictor. use new predictor for every time you run a technique
    :return: plots and saves confusion table
    '''
    resultColumnName = pr.resultColumn
    y_true = data[resultColumnName].astype(str)
    y_pred = pr.predict(data).astype(str)
    np.array(y_true).astype(str)
    np.array(y_pred).astype(str)

    np.set_printoptions(precision=2)
    plot_confusion_table(y_true=y_true.values, y_pred=y_pred, classes=pr._classes_, )

def plot_confusion_table(y_true, y_pred, classes):
    #fill confusionTable
    #for in case y_true and y_pred aren't string-arrays, convert them
    np.array(y_true).astype(str)
    np.array(y_pred).astype(str)
    np.array(classes).astype(str)

    columns = ["Times predicted",
               "True positive",
               "False positive",
               "False negative",
               "Sensitivity/Recall",
               "Specificity",
               "Accuracy",
               "Precision",
               "F-measure"
               ]

    #fill confusion Table with metrics

    classesWithsum = np.append(classes, "Sum")
    confusionTable = pd.DataFrame(index=classesWithsum, columns=columns).fillna(0.0)
    for rowIndex in range(len(y_true)):
        if y_pred[rowIndex] == y_true[rowIndex]:
            confusionTable.loc[y_pred[rowIndex]]["True positive"] += 1
        else:
            confusionTable.loc[y_pred[rowIndex]]["False positive"] += 1
            confusionTable.loc[y_true[rowIndex]]["False negative"] += 1
        confusionTable.loc[y_pred[rowIndex]]["Times predicted"] += 1
    #sum up
    sumColumns=["Times predicted", "True positive", "False positive", "False negative",]
    for column in sumColumns:
        confusionTable.loc["Sum"][column] = sum(confusionTable.loc[:][column])
    print("confusiontable predictions filled")

    #add metrics
    amountPredictions = len(y_true)
    for (idx, row) in confusionTable.iterrows():
        TP = row.loc["True positive"]
        FP = row.loc["False positive"]
        FN = row.loc["False negative"]
        TN = amountPredictions-TP-FP-FN

        precision = 0.0
        recall = 0.0
        if TP != 0.0 or FP != 0.0:
            precision = TP / (TP + FP)
        if TP != 0.0 or FN != 0.0:
            recall = TP / (TP + FN)
            row.loc["Sensitivity/Recall"] = recall
        if FP != 0.0 or TN != 0.0:
            row.loc["Specificity"] = TN/(FP+TN)
        row.loc["Precision"] = precision
        if recall != 0.0 or precision != 0.0:
            row.loc["F-measure"] = 2 * recall * precision / (recall + precision)
        if TP != 0.0 or TN != 0.0 or FP != 0.0:
            row.loc["Accuracy"] = (TP+TN)/(TP + TN + FP)
    avgColumns = ["Sensitivity/Recall", "Specificity", "Accuracy", "Precision", "F-measure"]
    for column in avgColumns:
        confusionTable.loc["Sum"][column] = confusionTable.loc[:][column].iloc[:-1].mean()
    print("confusiontable totally filled")
    confusionTable = confusionTable.round(5)
    intColumns = ["Times predicted",
               "True positive",
               "False positive",
               "False negative"]
    for column in confusionTable.columns:
        if column in intColumns:
            confusionTable[column]=confusionTable[column].astype(int)

    convert_dict = {"Times predicted": int,
                    "True positive": int,
                    "False positive": int,
                    "False negative": int,
                    "Sensitivity/Recall": float,
                    "Specificity": float,
                    "Accuracy": float,
                    "Precision": float,
                    "F-measure": float
                    }
    confusionTable = confusionTable.astype(convert_dict)
    # Normalize data to [0, 1] range for color mapping below
    colorMatrix = coloring(confusionTable)
    print("colortable filled")

    fig, ax = plt.subplots(1,1, figsize=(10,colorMatrix.__len__()/3), dpi=500)
    table = ax.table(cellText=confusionTable.values, rowLabels=confusionTable.index, colLabels=confusionTable.columns,
                         loc='center', cellColours=plt.cm.RdYlGn(colorMatrix), colWidths=[0.085 for x in confusionTable.columns] )
    table.auto_set_font_size(False)
    table.set_fontsize(6)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    plt.gcf().subplots_adjust( left=0.005, right =0.0051)
    print("table ready to show")
    plt.show()
    print("Accuracytable ploted")
    save("Accuracytable" , fig=fig, plt=plt)

def coloring(confusionTable):
    colorMatrix = pd.DataFrame(index=confusionTable.index, columns=confusionTable.columns).fillna(0.5)
    redishColumns = ["False positive", "False negative"]
    greenishColumns = ["Times predicted", "True positive"]
    redishIfNotOneColumns = ["Sensitivity/Recall", "Specificity", "Precision"]
    for column in confusionTable.columns:
        for rowIndex in range(len(confusionTable)):
            columnMax = max(confusionTable.iloc[0:-1].loc[:][column])
            if confusionTable.iloc[rowIndex].loc[column] != 0 and columnMax > 0 and (column != redishIfNotOneColumns or confusionTable.iloc[rowIndex].loc[column] != 1):
                if column in greenishColumns:
                    colorMatrix.iloc[rowIndex].loc[column] = 0.5 + (confusionTable.iloc[rowIndex].loc[column] / columnMax*1.5)
                if column in redishColumns:
                    colorMatrix.iloc[rowIndex].loc[column] = 0.5 - (confusionTable.iloc[rowIndex].loc[column] / columnMax * 1.5)
                if column in redishIfNotOneColumns:
                    colorMatrix.iloc[rowIndex].loc[column] = 0.5 - (1-confusionTable.iloc[rowIndex].loc[column] * 1.5)
    return colorMatrix
