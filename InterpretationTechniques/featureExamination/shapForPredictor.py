from Connections.predictor import *
import shap         #to install SHAP-package, install C++ Build tools first: https://go.microsoft.com/fwlink/?LinkId=691126
import matplotlib.pyplot as plt
from InterpretationTechniques.PlotAndShow import *
# code retrieved from: https://github.com/slundberg/shap and https://slundberg.github.io/shap/notebooks/Census%20income%20classification%20with%20scikit-learn.html
def plotShap(data, resultColumnName, columnsThatAreRoundedToTwoDecimals = None, pr = Predictor(returnDistanceOfClass=True, callingFunction="shap")):
    shap.initjs()
    data[columnsThatAreRoundedToTwoDecimals] = data[columnsThatAreRoundedToTwoDecimals].astype(float).round(2).astype(str)
    #initiate predictor and encode data
    dataEn = pr.encode(data).astype(float)
    dataEn = dataEn.set_index(dataEn[resultColumnName].values)

    #seperate data
    slicingIndex = int(len(data)/2)
    X = dataEn.iloc[slicingIndex:]
    X_train = dataEn.iloc[:slicingIndex]

    # initiate explainer
    med = X_train.median().values.reshape((1, X_train.shape[1])).astype(int)
    med = pd.DataFrame(med, index = [med[0,1]], columns=data.columns)
    explainer = shap.KernelExplainer(pr.predict, X_train, keep_index=True)

    #single prediction
    rowIndex = 9
    shap_values_single = explainer.shap_values(X.iloc[[rowIndex]], nsamples=1000)
    shap.force_plot(explainer.expected_value, shap_values_single, data.iloc[rowIndex, :])

    #global predictions
    shap_values = explainer.shap_values(X, nsamples=1000)
    shap.force_plot(explainer.expected_value, shap_values, data)
    for column in X.columns:
        shap.dependence_plot(column, shap_values, X, interaction_index=resultColumnName,display_features=data)

    shap.summary_plot(shap_values, X, title="Summary Plot SHAP")
    shap.summary_plot(shap_values, X, plot_type="bar")

    return shap_values, explainer, data, X, X_train


