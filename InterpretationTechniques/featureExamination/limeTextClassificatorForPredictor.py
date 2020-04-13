from eli5.lime import TextExplainer
import eli5 as eli5
import matplotlib.pyplot as plt
from Connections.predictor import *
from IPython.core.interactiveshell import InteractiveShell

def limeTextClassification(dataset,
                           data,   pr = Predictor(callingFunction = "TextClassifier")):  # example retrieved from https://eli5.readthedocs.io/en/latest/tutorials/black-box-text-classifiers.html#textexplainer

    pr.dataset = dataset
    resultColumnName = pr.resultColumn
    dataClasses = list(dict.fromkeys(data[resultColumnName].astype(str)))
    dataClasses.sort()

    te = TextExplainer(random_state=42)

def limeTextClassification(dataset,
                               data, pr=Predictor(
                callingFunction="TextClassifier")):  # example retrieved from https://eli5.readthedocs.io/en/latest/tutorials/black-box-text-classifiers.html#textexplainer

    pr = Predictor(dataset=dataset, callingFunction="TextClassifier")
    resultColumnName = pr.resultColumn
    dataClasses = list(dict.fromkeys(data[resultColumnName].astype(str)))
    dataClasses.sort()
    pr = Predictor(dataset=dataset, callingFunction="TextClassifier")

    te = TextExplainer(random_state=42)
    te.fit(dataset["text"], pr.predict_proba)

    te.fit(dataset["text"], pr.predict_proba)
    te.show_prediction(target_names=pr._classes_000.tolist())

    return te, pr._classes_000.tolist()
    '''
    te.explain_prediction()
    te.show_weights()
    te.explain_weights()
    print(te.metrics_)
    '''
