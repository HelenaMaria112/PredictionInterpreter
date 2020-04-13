from Connections.predictor import *
import copy
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from InterpretationTechniques.PlotAndShow import *

def plotPrecisionRecall(data):
    pr = Predictor()
    y_test = data["gknto"]
    #y_score = pr.predict(data, "valueDistance")

    #y_score and Y_test have strucutre: n_datasets, n_classes
    y_score = pr.predict_proba_ErrorMatrix(data)
    Y_test = fillTrueMatrix(data["gknto"], y_score)
    y_score, Y_test = deleteNotUsedColumns(y_score, Y_test)
    print('columns after reduction: ' + str(len(y_score.columns)))
    classes=y_score.columns

    # all the underneith is retrieved and adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html !!Multiclass!!
    # For each class

    n_classes = len(classes)
    y_score= y_score.values
    Y_test = Y_test.values
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
                                                                    y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))
    #plot average precision score vs recall
    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
             where='post')
    #plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
            .format(average_precision["micro"]))
    plt.show()
    save("Precision ", plt=plt)

    #plot precision over all classes
    from itertools import cycle
    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(20, 20))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(classes[i], average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, prop=dict(size=14), loc='upper left', bbox_to_anchor=(1, 1))

    save("Extension of Precision-Recall curve to multi-class ", fig=fig, plt=plt)

def fillTrueMatrix(y_data,y_score):
    y_test = copy.deepcopy(y_score)
    y_test[:] = 0.0
    for rowIndex in range(len(y_data)):
        y_test.iloc[rowIndex].loc[str(y_data.iloc[rowIndex])] = 1.0
    return y_test

def deleteNotUsedColumns(y_score, Y_test):
    print('columns before reduction: ' + str(len(y_score.columns)))
    classes = y_score.columns
    for columnIndex in range(len(classes)):
        columnName = classes[columnIndex]
        #if there is less than 5 predictions/trueValues for a class, delete class
        if (y_score[columnName].isin([0.0]).sum() > (len(y_score) - 5) or Y_test[columnName].isin([0.0]).sum() > (len(y_score) - 5)):   #alternatively: y_score[columnName]==0).all() and (Y_test[columnName]==0).all()  #all returns false, if all values are zero
            y_score = y_score.drop([columnName], axis=1)
            Y_test = Y_test.drop([columnName], axis=1)
            print('columns after reduction: ' + str(len(y_score.columns)))
    print('columns after total reduction: ' + str(len(y_score.columns)))
    return y_score, Y_test